/**
 * WebGPU pipeline and buffer management utilities.
 *
 * Handles shader compilation, compute pipeline creation,
 * buffer allocation, and data transfer.
 */

import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SHADER_DIR = join(__dirname, '..', 'shaders');

/**
 * Load a WGSL shader source file.
 * @param {string} name - Shader filename (e.g. 'logmath.wgsl')
 * @returns {string} WGSL source code
 */
export function loadShader(name) {
  return readFileSync(join(SHADER_DIR, name), 'utf8');
}

/**
 * Create a compute pipeline from WGSL source with overrides.
 *
 * @param {GPUDevice} device
 * @param {string} wgslSource - WGSL shader source
 * @param {Object<string,number>} [constants={}] - Override constants
 * @param {string} [entryPoint='main']
 * @returns {GPUComputePipeline}
 */
export function createPipeline(device, wgslSource, constants = {}, entryPoint = 'main') {
  const module = device.createShaderModule({ code: wgslSource });
  return device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
      entryPoint,
      constants,
    },
  });
}

/**
 * Create a GPU storage buffer from a typed array.
 *
 * @param {GPUDevice} device
 * @param {Float32Array|Uint32Array} data
 * @param {number} [usage] - GPUBufferUsage flags
 * @returns {GPUBuffer}
 */
export function createStorageBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: usage || (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
    mappedAtCreation: true,
  });
  const mapped = new (data.constructor)(buffer.getMappedRange());
  mapped.set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create an empty GPU storage buffer.
 *
 * @param {GPUDevice} device
 * @param {number} byteSize
 * @param {number} [usage]
 * @returns {GPUBuffer}
 */
export function createEmptyBuffer(device, byteSize, usage) {
  return device.createBuffer({
    size: byteSize,
    usage: usage || (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
  });
}

/**
 * Create a uniform buffer from data.
 *
 * @param {GPUDevice} device
 * @param {ArrayBuffer|TypedArray} data
 * @returns {GPUBuffer}
 */
export function createUniformBuffer(device, data) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer || data));
  buffer.unmap();
  return buffer;
}

/**
 * Read data back from a GPU buffer.
 *
 * @param {GPUDevice} device
 * @param {GPUBuffer} buffer
 * @param {number} byteSize
 * @returns {Promise<Float32Array>}
 */
export async function readBuffer(device, buffer, byteSize) {
  const readBuffer = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, byteSize);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  return data;
}

/**
 * Create a bind group for a compute pipeline.
 *
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {number} groupIndex
 * @param {GPUBuffer[]} buffers
 * @returns {GPUBindGroup}
 */
export function createBindGroup(device, pipeline, groupIndex, buffers) {
  const entries = buffers.map((buffer, i) => ({
    binding: i,
    resource: { buffer },
  }));
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(groupIndex),
    entries,
  });
}

/**
 * Dispatch a compute shader.
 *
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup} bindGroup
 * @param {number} workgroupCountX
 * @param {number} [workgroupCountY=1]
 * @param {number} [workgroupCountZ=1]
 */
export function dispatch(device, pipeline, bindGroup, workgroupCountX, workgroupCountY = 1, workgroupCountZ = 1) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Run a sequence of dispatches, submitting them all at once.
 *
 * @param {GPUDevice} device
 * @param {Array<{pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number, number?, number?]}>} dispatches
 */
export function batchDispatch(device, dispatches) {
  const encoder = device.createCommandEncoder();
  for (const { pipeline, bindGroup, workgroups } of dispatches) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
}
