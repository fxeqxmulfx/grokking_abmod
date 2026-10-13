import type {Weights, NetworkView, WeightMode} from './types.ts';
export type TopologyRequest =
  | {type: 'initialize'; canvas: OffscreenCanvas}
  | {type: 'clear'; job: number}
  | {type: 'draw'; job: number; width: number; height: number; channels: number; scale: number; view: NetworkView; selected: number; focus: number; highlight: boolean; mode: WeightMode; weights?: Weights; initial?: Weights};
export type TopologyResponse = {type: 'paint'; job: number; pathBuilds: number; backgroundPaints: number; buildMs: number; drawMs: number} | {type: 'error'; message: string};
