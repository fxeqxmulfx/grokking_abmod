import {errorText} from './types.ts';
import type {Weights} from './types.ts';
import {CPUScene, topologyLayout} from './topology-scene.ts';
import type {TopologyRequest, TopologyResponse} from './topology-protocol.ts';
declare function postMessage(message: TopologyResponse): void;
let canvas: OffscreenCanvas, ctx: OffscreenCanvasRenderingContext2D;
let weights: Weights, initial: Weights, pending: Exclude<TopologyRequest, {type: 'initialize'}> | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;
const ready = CPUScene.create();
onmessage = ({data}: MessageEvent<TopologyRequest>) => {
  try {
    if (data.type === 'initialize') {
      canvas = data.canvas; const context = canvas.getContext('2d', {willReadFrequently: true});
      if (!context) throw new Error('Canvas 2D is unavailable.'); ctx = context; return;
    }
    // Retain snapshots even when a newer pose replaces a queued draw.
    if (data.type === 'draw') {if (data.weights) weights = data.weights; if (data.initial) initial = data.initial;}
    pending = data;
    if (timer !== null) return;
    timer = setTimeout(async () => {
      try {
        const scene = await ready;
        timer = null; const request = pending!; pending = null;
        if (request.type === 'clear') {ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);}
        else {
          const width = Math.round(request.width * request.scale), height = Math.round(request.height * request.scale);
          if (canvas.width !== width) canvas.width = width; if (canvas.height !== height) canvas.height = height;
          scene.draw(ctx, topologyLayout(request.width, request.height, request.channels), weights, request.selected, request.view, request.focus, {initial, mode: request.mode, highlight: request.highlight}, request.scale);
        }
        postMessage({type: 'paint', job: request.job, pathBuilds: scene.pathBuilds, backgroundPaints: scene.backgroundPaints, buildMs: scene.buildMs, drawMs: scene.drawMs});
      } catch (error) {timer = null; postMessage({type: 'error', message: errorText(error)});}
    }, 0);
  } catch (error) {postMessage({type: 'error', message: errorText(error)});}
};
