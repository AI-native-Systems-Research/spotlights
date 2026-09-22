/**
 * Shared path constants derived from the single source of truth (storyboard.mjs).
 * Side-effect free: safe to import anywhere.
 */

import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';
import { PAGE_RELATIVE_PATH } from './storyboard.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = resolve(HERE, '..', '..');
export const PAGE_URL = pathToFileURL(resolve(REPO_ROOT, PAGE_RELATIVE_PATH)).href;
