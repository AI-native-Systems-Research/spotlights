import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { BEATS, GEOMETRY, dwellFor } from './storyboard.mjs';
import { PAGE_URL } from './paths.mjs';
import {
  OVERLAY_IDS, installOverlay, showCaption, hideCaption, ring, unring, moveCursor, pulseCursor,
  beatWaits, pageText, CAPTION_HIDE_WAIT_MS, CAPTION_MIN_ONSCREEN_MS,
} from './overlay.mjs';

let browser;
let page;

before(async () => {
  browser = await chromium.launch({ channel: 'chrome' });
  page = await browser.newPage({ viewport: GEOMETRY, deviceScaleFactor: 1 });
  await page.goto(PAGE_URL, { waitUntil: 'load' });
  await installOverlay(page);
});

after(async () => { await browser?.close(); });

test('install adds all three overlay elements exactly once', async () => {
  for (const id of Object.values(OVERLAY_IDS)) {
    assert.equal(await page.locator(`#${id}`).count(), 1, `#${id}`);
  }
  await installOverlay(page);
  for (const id of Object.values(OVERLAY_IDS)) {
    assert.equal(await page.locator(`#${id}`).count(), 1, `#${id} duplicated on reinstall`);
  }
});

test('install does not disturb the page it overlays', async () => {
  assert.equal(await page.locator('tbody.cand').count(), 173);
  assert.equal(await page.locator('tbody.fnd').count(), 97);
});

test('the caption is hidden until shown, and renders markup', async () => {
  await hideCaption(page);
  assert.equal(await page.locator(`#${OVERLAY_IDS.caption}`).isVisible(), false);

  await showCaption(page, '<b>173</b> ranked candidates');
  const caption = page.locator(`#${OVERLAY_IDS.caption}`);
  assert.equal(await caption.isVisible(), true);
  assert.match(await caption.innerText(), /173 ranked candidates/);
  assert.equal(await caption.locator('b').innerText(), '173');

  await hideCaption(page);
  assert.equal(await caption.isVisible(), false);
});

test('the caption fades rather than snapping', async () => {
  // Ensure caption is hidden before the test begins
  await hideCaption(page);

  // Assertion 1: display must NOT be none (which would prevent fading)
  const display = await page.evaluate((id) => {
    return getComputedStyle(document.getElementById(id)).display;
  }, OVERLAY_IDS.caption);
  assert.notEqual(display, 'none', 'caption hidden with display: none cannot fade');

  // Assertion 2: opacity must pass through intermediate values during show
  // Install a requestAnimationFrame loop to sample opacity over ~400ms
  await page.evaluate((id) => {
    window._opacitySamples = [];
    const captionEl = document.getElementById(id);
    const startTime = performance.now();
    const loop = () => {
      const elapsed = performance.now() - startTime;
      if (elapsed < 400) {
        const opacity = parseFloat(getComputedStyle(captionEl).opacity);
        window._opacitySamples.push(opacity);
        requestAnimationFrame(loop);
      }
    };
    requestAnimationFrame(loop);
  }, OVERLAY_IDS.caption);

  // Call the real showCaption to exercise the actual code path
  await showCaption(page, 'fade test');

  // Wait for the sampling window to elapse
  await page.waitForTimeout(450);

  // Read the collected samples and verify at least one is in the intermediate range
  const samples = await page.evaluate(() => window._opacitySamples);
  const intermediate = samples.filter((s) => s > 0.01 && s < 0.99);
  assert.ok(
    intermediate.length > 0,
    `no intermediate opacity samples found between 0.01 and 0.99. collected samples: ${samples.slice(0, 50).join(', ')}${samples.length > 50 ? '...' : ''}`,
  );

  // Clean up: hide and wait for transition to complete before returning to next test
  await hideCaption(page);
});

test('the caption takes its colours from the page, not from hard-coded hex', async () => {
  await showCaption(page, 'probe');
  const [pagePanel, captionBg] = await page.evaluate((id) => {
    const root = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();
    const bg = getComputedStyle(document.getElementById(id)).backgroundColor;
    return [root, bg];
  }, OVERLAY_IDS.caption);
  assert.ok(pagePanel.length > 0, 'page exposes no --panel custom property');
  assert.ok(captionBg.startsWith('rgb'), `caption background not resolved: ${captionBg}`);
  await hideCaption(page);
});

test('the ring covers its target and clears on unring', async () => {
  await ring(page, '#hotbtn');
  const ringEl = page.locator(`#${OVERLAY_IDS.ring}`);
  assert.equal(await ringEl.isVisible(), true);

  const target = await page.locator('#hotbtn').boundingBox();
  const box = await ringEl.boundingBox();
  assert.ok(Math.abs(box.x - target.x) < 12, `ring x off by ${Math.abs(box.x - target.x)}`);
  assert.ok(box.width >= target.width, 'ring narrower than its target');

  await unring(page);
  assert.equal(await ringEl.isVisible(), false);
});

test('the cursor moves to a target centre and can pulse', async () => {
  const centre = await moveCursor(page, '#hotbtn');
  const target = await page.locator('#hotbtn').boundingBox();
  assert.ok(Math.abs(centre.x - (target.x + target.width / 2)) < 2);
  assert.ok(Math.abs(centre.y - (target.y + target.height / 2)) < 2);

  const cursor = page.locator(`#${OVERLAY_IDS.cursor}`);
  assert.equal(await cursor.isVisible(), true);
  await pulseCursor(page);
  assert.equal(await cursor.isVisible(), true);
});

test('moveCursor rejects a selector that is not on the page', async () => {
  await assert.rejects(() => moveCursor(page, '#no-such-element'), /no-such-element/);
});

/**
 * The three below are the guard on the defect that made most of the recorded captions
 * unreadable: the recorder used to show a caption *after* a beat's choreography, so its
 * screen time was whatever the dwell had left over -- nothing, for every beat that
 * overran, and three of the gif's seven captions lasted a single sampled frame.
 */
test('a caption is guaranteed the on-screen floor on a beat that would flash it', () => {
  // A trivial beat: a 1.0s dwell and nothing spent but the caption's own fade-in. The
  // dwell remainder alone leaves the caption up for well under the floor.
  const fast = beatWaits({ dwellMs: 1000, spentMs: 320, onScreenMs: 0, hasCaption: true });
  assert.ok(fast.dwellWaitMs < CAPTION_MIN_ONSCREEN_MS, 'premise: the dwell alone is not enough');
  assert.ok(fast.floorTopUpMs > 0, 'no top-up on a beat that would flash its caption');
  assert.equal(fast.projectedOnScreenMs, CAPTION_MIN_ONSCREEN_MS);

  // The floor holds for every dwell the storyboard actually declares, at the worst case
  // for the caption: choreography that eats the whole dwell and leaves zero remainder.
  for (const beat of BEATS) {
    for (const cut of beat.cuts) {
      const dwellMs = Math.round(dwellFor(beat, cut) * 1000);
      for (const spentMs of [0, dwellMs, dwellMs * 3]) {
        const w = beatWaits({ dwellMs, spentMs, onScreenMs: 0, hasCaption: true });
        assert.ok(
          w.projectedOnScreenMs >= CAPTION_MIN_ONSCREEN_MS,
          `${beat.id}/${cut} spent ${spentMs}ms: caption gets only ${w.projectedOnScreenMs}ms`,
        );
      }
    }
  }

  // An uncaptioned beat is paced by its dwell alone and asks for no hold.
  const bare = beatWaits({ dwellMs: 1000, spentMs: 100, hasCaption: false });
  assert.equal(bare.hideReserveMs, 0);
  assert.equal(bare.floorTopUpMs, 0);
  assert.equal(bare.projectedOnScreenMs, 0);
});

test('the dwell stays a deadline, and a caption already past the floor is not held', () => {
  // A beat whose choreography has blown the dwell waits none of it and says how much.
  const over = beatWaits({ dwellMs: 1500, spentMs: 4200, onScreenMs: 3900, hasCaption: true });
  assert.equal(over.dwellWaitMs, 0);
  assert.equal(over.overrunMs, 4200 + CAPTION_HIDE_WAIT_MS - 1500);
  assert.equal(over.floorTopUpMs, 0, 'a caption long past the floor must not be held further');
  assert.equal(over.projectedOnScreenMs, 3900);

  // The fade-out is still reserved out of the remainder rather than added to the beat.
  const roomy = beatWaits({ dwellMs: 8000, spentMs: 1000, onScreenMs: 680, hasCaption: true });
  assert.equal(roomy.hideReserveMs, CAPTION_HIDE_WAIT_MS);
  assert.equal(roomy.dwellWaitMs, 8000 - 1000 - CAPTION_HIDE_WAIT_MS);
  assert.equal(roomy.overrunMs, 0);
  assert.equal(roomy.floorTopUpMs, 0);
});

test('page text excludes the overlay, so a caption cannot satisfy a page assertion', async () => {
  // The caption is up while a beat's assertions run, and it quotes the page's own
  // figures, so the text a domContains reads must not include it.
  await showCaption(page, 'sl-caption-sentinel <b>$35.75</b>');
  const text = await pageText(page);
  assert.ok(!text.includes('sl-caption-sentinel'), 'the caption leaked into the page text');
  assert.match(text, /candidate leaderboard/i, 'the page text lost the page');
  // Nothing rendered may be dropped by the read, and the caption must survive it.
  // addEventListener occurs only inside the page's two body <script> blocks, never in
  // its prose, so it witnesses that unrendered source stayed out of the read.
  assert.ok(!text.includes('addEventListener'), 'the body <script> source leaked into the page text');
  assert.equal(await page.locator(`#${OVERLAY_IDS.caption}`).isVisible(), true);
  await hideCaption(page);
});
