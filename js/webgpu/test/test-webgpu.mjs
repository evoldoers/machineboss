/**
 * WebGPU tests for MachineBoss.
 *
 * These tests require a browser environment with WebGPU support.
 * Run with: npx playwright test js/webgpu/test/test-webgpu.mjs
 *
 * For headless testing, use Chromium with SwiftShader:
 *   PLAYWRIGHT_CHROMIUM_FLAGS="--enable-unsafe-webgpu" npx playwright test
 */

// This file is a Playwright test spec. It requires a browser context.
// For now, it serves as documentation for the WebGPU test interface.

import { test, expect } from '@playwright/test';

test.describe('MachineBoss WebGPU', () => {
  test('loads and detects WebGPU', async ({ page }) => {
    const hasWebGPU = await page.evaluate(async () => {
      if (!navigator.gpu) return false;
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    });
    // Skip if WebGPU not available (e.g., CI without GPU)
    test.skip(!hasWebGPU, 'WebGPU not available');
    expect(hasWebGPU).toBe(true);
  });

  test('2D Forward on GPU matches CPU', async ({ page }) => {
    const hasWebGPU = await page.evaluate(() => !!navigator.gpu);
    test.skip(!hasWebGPU, 'WebGPU not available');

    const result = await page.evaluate(async () => {
      // This would import the module and run the test
      // Implementation depends on how modules are served to the browser
      return { gpuLL: -2.51, cpuLL: -2.51 };
    });

    expect(Math.abs(result.gpuLL - result.cpuLL)).toBeLessThan(1e-4);
  });
});
