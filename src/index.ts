// Facade
export { O3DGraph } from './O3DGraph.js';
export type { O3DGraphDeps, FrameScheduler, GraphRenderSource, MutationOptions } from './O3DGraph.js';

// Contracts & public types
export * from './types/index.js';

// Events
export { EventBus } from './events/EventBus.js';
export type { Listener } from './events/EventBus.js';

// Renderer
export { O3DEngineAdapter, O3DRendererAdapter } from './renderer/O3DRendererAdapter.js';
export type { O3DWasmBindings, O3DRendererOptions } from './renderer/O3DRendererAdapter.js';
export { O3DBatchRenderer } from './renderer/O3DBatchRenderer.js';
export type {
  O3DBatchRendererOptions,
  O3DCameraControllerLike,
  O3DGpuMeshLike,
  O3DInstancedMeshLike,
  O3DLineBatchLike,
} from './renderer/O3DBatchRenderer.js';

// Memory
export { GraphStore, DEFAULT_STORE_OPTIONS } from './memory/GraphStore.js';

// Spatial
export { SpatialTree, SPATIAL_TREE_MAX_DEPTH } from './spatial/SpatialTree.js';

// Physics
export { PhysicsSimulator, resolvePhysicsOptions } from './physics/PhysicsSimulator.js';
export type { PhysicsSimulatorDeps, ResolvedPhysicsOptions } from './physics/PhysicsSimulator.js';
export {
  CenterForce,
  CollideForce,
  ContainForce,
  LinkForce,
  ManyBodyForce,
  RadialForce,
  createDefaultForces,
} from './physics/forces/index.js';
export type {
  CenterForceOptions,
  CollideForceOptions,
  ContainForceOptions,
  LinkForceOptions,
  ManyBodyForceOptions,
  RadialForceOptions,
} from './physics/forces/index.js';

// One-call view on @omernach/o3d
export { createGraphView } from './view/createGraphView.js';
export type { FocusOptions, GraphView, GraphViewOptions, O3DModule, ViewInsets } from './view/createGraphView.js';

// Camera
export { CameraRig } from './camera/CameraRig.js';
export type { CameraRigCamera, CameraRigOptions, Vec3 } from './camera/CameraRig.js';

// Labels
export { LabelLayer } from './labels/LabelLayer.js';
export type { EdgeLabelMode, LabelLayerOptions, LabelSource } from './labels/LabelLayer.js';

// Interaction
export { InteractionController, resolveInteractionOptions } from './interaction/InteractionController.js';
export type { InteractionControllerDeps, ResolvedInteractionOptions } from './interaction/InteractionController.js';

// Utilities
export {
  encodeColorLayout,
  packRGBA,
  packRGBAf,
  parseColor,
  unpackA,
  unpackB,
  unpackG,
  unpackR,
  withAlpha,
} from './utils/color.js';
export { Lcg } from './utils/random.js';
