/**
 * The three things drawn on top of the report while recording.
 *
 * Colour comes only from the page's own CSS custom properties, so the caption
 * card matches the report by construction rather than by eye. Playwright's video
 * output renders no mouse pointer, which is why the cursor here is synthetic:
 * without it, filter and toggle changes look like they happen by themselves.
 */

export const OVERLAY_IDS = {
  caption: 'sl-demo-caption',
  cursor: 'sl-demo-cursor',
  ring: 'sl-demo-ring',
};

const STYLE_ID = 'sl-demo-overlay-style';

// Transition durations from CSS (in milliseconds). Hide helpers wait these + ~60ms slack.
const CAPTION_TRANSITION_MS = 280;
const RING_TRANSITION_MS = 250;
const HIDE_SLACK_MS = 60;

// Total wait times for hide operations, derived from transition + slack above.
// Must exceed the actual transition time to avoid racing the CSS visibility flip.
const CAPTION_HIDE_WAIT_MS = CAPTION_TRANSITION_MS + HIDE_SLACK_MS;
const RING_HIDE_WAIT_MS = RING_TRANSITION_MS + HIDE_SLACK_MS;

const CSS = `
#${OVERLAY_IDS.caption} {
  position: fixed; left: 50%; bottom: 56px; transform: translateX(-50%) translateY(8px);
  max-width: 1040px; padding: 16px 24px 16px 20px;
  background: var(--panel); color: var(--ink);
  border: 1px solid var(--border-strong); border-left: 4px solid var(--accent);
  border-radius: 12px; box-shadow: var(--shadow);
  font-family: var(--sans); font-size: 26px; line-height: 1.35; letter-spacing: -0.01em;
  opacity: 0; transition: opacity .28s ease, transform .28s ease, visibility .28s ease;
  z-index: 2147483000; pointer-events: none; visibility: hidden;
}
#${OVERLAY_IDS.caption}[data-shown="1"] { opacity: 1; transform: translateX(-50%) translateY(0); visibility: visible; }
#${OVERLAY_IDS.caption} b { color: var(--hi); font-weight: 700; }
#${OVERLAY_IDS.caption} code {
  font-family: var(--mono); font-size: .88em;
  background: var(--panel-2); padding: 1px 6px; border-radius: 5px;
}
#${OVERLAY_IDS.cursor} {
  position: fixed; top: 0; left: 0; width: 18px; height: 18px; margin: -9px 0 0 -9px;
  border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow), 0 2px 6px rgba(0,0,0,.35);
  opacity: 0; transition: transform .45s cubic-bezier(.22,.61,.36,1), opacity .2s ease;
  z-index: 2147483001; pointer-events: none;
}
#${OVERLAY_IDS.cursor}[data-shown="1"] { opacity: 1; }
#${OVERLAY_IDS.cursor}[data-pulse="1"] {
  animation: sl-demo-pulse .45s ease-out;
}
@keyframes sl-demo-pulse {
  0%   { box-shadow: 0 0 0 3px var(--accent-glow); }
  60%  { box-shadow: 0 0 0 20px rgba(216,145,42,0); }
  100% { box-shadow: 0 0 0 3px var(--accent-glow); }
}
#${OVERLAY_IDS.ring} {
  position: fixed; border: 2px solid var(--accent); border-radius: 10px;
  box-shadow: 0 0 0 6px var(--accent-glow); opacity: 0; transition: opacity .25s ease, visibility .25s ease;
  z-index: 2147482999; pointer-events: none; visibility: hidden;
}
#${OVERLAY_IDS.ring}[data-shown="1"] { opacity: 1; visibility: visible; }
`;

export async function installOverlay(page) {
  await page.evaluate(
    ({ ids, styleId, css }) => {
      if (!document.getElementById(styleId)) {
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = css;
        document.head.appendChild(style);
      }
      for (const id of Object.values(ids)) {
        if (!document.getElementById(id)) {
          const el = document.createElement('div');
          el.id = id;
          document.body.appendChild(el);
        }
      }
    },
    { ids: OVERLAY_IDS, styleId: STYLE_ID, css: CSS },
  );
}

export async function showCaption(page, html) {
  await page.evaluate(
    ({ id, html: markup }) => {
      const el = document.getElementById(id);
      el.innerHTML = markup;
      el.dataset.shown = '1';
    },
    { id: OVERLAY_IDS.caption, html },
  );
  await page.waitForTimeout(320);
}

export async function hideCaption(page) {
  await page.evaluate((id) => {
    document.getElementById(id).dataset.shown = '0';
  }, OVERLAY_IDS.caption);
  await page.waitForTimeout(CAPTION_HIDE_WAIT_MS);
}

export async function ring(page, selector) {
  const box = await boxOf(page, selector);
  await page.evaluate(
    ({ id, box: b }) => {
      const el = document.getElementById(id);
      const pad = 6;
      el.style.left = `${b.x - pad}px`;
      el.style.top = `${b.y - pad}px`;
      el.style.width = `${b.width + pad * 2}px`;
      el.style.height = `${b.height + pad * 2}px`;
      el.dataset.shown = '1';
    },
    { id: OVERLAY_IDS.ring, box },
  );
  await page.waitForTimeout(280);
}

export async function unring(page) {
  await page.evaluate((id) => {
    document.getElementById(id).dataset.shown = '0';
  }, OVERLAY_IDS.ring);
  await page.waitForTimeout(RING_HIDE_WAIT_MS);
}

export async function moveCursor(page, selector) {
  const box = await boxOf(page, selector);
  const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await page.evaluate(
    ({ id, point }) => {
      const el = document.getElementById(id);
      el.dataset.shown = '1';
      el.style.transform = `translate(${point.x}px, ${point.y}px)`;
    },
    { id: OVERLAY_IDS.cursor, point: centre },
  );
  await page.waitForTimeout(480);
  return centre;
}

export async function pulseCursor(page) {
  await page.evaluate((id) => {
    const el = document.getElementById(id);
    el.dataset.pulse = '0';
    void el.offsetWidth;
    el.dataset.pulse = '1';
  }, OVERLAY_IDS.cursor);
  await page.waitForTimeout(460);
}

async function boxOf(page, selector) {
  let box;
  try {
    box = await page.locator(selector).first().boundingBox({ timeout: 2000 });
  } catch (err) {
    throw new Error(`cannot locate ${selector} to position an overlay`);
  }
  if (!box) throw new Error(`cannot locate ${selector} to position an overlay`);
  return box;
}
