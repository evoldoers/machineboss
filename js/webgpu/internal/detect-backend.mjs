/**
 * WebGPU feature detection and backend selection.
 */

/**
 * Detect whether WebGPU is available and return a GPUDevice if so.
 * @returns {Promise<{backend: 'webgpu'|'cpu', device: GPUDevice|null}>}
 */
export async function detectBackend() {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        return { backend: 'webgpu', device };
      }
    } catch (e) {
      // WebGPU not available, fall through
    }
  }
  return { backend: 'cpu', device: null };
}
