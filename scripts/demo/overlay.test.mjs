import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GEOMETRY, PAGE_RELATIVE_PATH } from './storyboard.mjs';
import {
  OVERLAY_IDS, installOverlay, showCaption, hideCaption, ring, unring, moveCursor, pulseCursor,
} from './overlay.mjs';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const PAGE_URL = pathToFileURL(resolve(REPO_ROOT, PAGE_RELATIVE_PATH)).href;

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
