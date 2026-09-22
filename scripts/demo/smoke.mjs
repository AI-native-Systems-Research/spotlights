// Proves Playwright can open the self-contained report over file:// with real fonts.
import { chromium } from 'playwright';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = resolve(HERE, '..', '..');
export const PAGE_URL = pathToFileURL(
  resolve(REPO_ROOT, 'examples/vllm_subset/experiment.html'),
).href;
export const GEOMETRY = { width: 1440, height: 810 };

const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: GEOMETRY, deviceScaleFactor: 1 });
await page.goto(PAGE_URL, { waitUntil: 'load' });

const counts = await page.evaluate(() => ({
  candidates: document.querySelectorAll('tbody.cand').length,
  findings: document.querySelectorAll('tbody.fnd').length,
  hero: document.querySelector('.hero .fig')?.textContent?.trim(),
  monoFont: getComputedStyle(document.querySelector('.mono')).fontFamily,
}));
console.log(JSON.stringify(counts, null, 2));

await mkdir(resolve(HERE, 'out'), { recursive: true });
await page.screenshot({ path: resolve(HERE, 'out/smoke.png') });
await browser.close();

if (counts.candidates !== 173 || counts.findings !== 97 || counts.hero !== '173') {
  throw new Error(`page did not load as expected: ${JSON.stringify(counts)}`);
}
console.log('smoke ok');
