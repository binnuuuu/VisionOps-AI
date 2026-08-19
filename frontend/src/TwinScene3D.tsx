import { useEffect, useRef } from "react";
import * as THREE from "three";
import { mediaUrl } from "./api";
import type { TwinSnapshot } from "./types";

type TwinScene3DProps = {
  snapshot: TwinSnapshot;
  operatorImageUrl?: string | null;
};

type RuntimeScene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  activeProduct: THREE.Group;
  queuedProducts: THREE.Mesh[];
  rollers: THREE.Mesh[];
  gatePivot: THREE.Group;
  scanPlane: THREE.Mesh;
  cameraLight: THREE.PointLight;
  cameraLensMaterial: THREE.MeshStandardMaterial;
  inspectionMaterial: THREE.MeshStandardMaterial;
  decisionHalo: THREE.Mesh;
  decisionHaloMaterial: THREE.MeshBasicMaterial;
  monitorMaterial: THREE.MeshBasicMaterial;
  frameId: number;
  observer: ResizeObserver;
};

function standardMaterial(color: number, metalness = 0.72, roughness = 0.3): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, metalness, roughness });
}

function box(
  width: number,
  height: number,
  depth: number,
  material: THREE.Material,
  x: number,
  y: number,
  z: number
): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function createProduct(): THREE.Group {
  const group = new THREE.Group();
  const carrier = box(1.05, 0.12, 0.72, standardMaterial(0x28302f, 0.65, 0.35), 0, 0, 0);
  group.add(carrier);

  const capsuleMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x38d26f,
    emissive: 0x092c17,
    metalness: 0.08,
    roughness: 0.2,
    transmission: 0.18,
    transparent: true,
    opacity: 0.96
  });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.52, 24), capsuleMaterial);
  body.rotation.z = Math.PI / 2;
  body.position.y = 0.22;
  body.castShadow = true;
  group.add(body);
  for (const x of [-0.26, 0.26]) {
    const end = new THREE.Mesh(new THREE.SphereGeometry(0.15, 20, 12), capsuleMaterial);
    end.position.set(x, 0.22, 0);
    end.castShadow = true;
    group.add(end);
  }
  return group;
}

function createBin(color: number, x: number, z: number): THREE.Group {
  const group = new THREE.Group();
  const material = standardMaterial(color, 0.55, 0.42);
  const sideMaterial = standardMaterial(0x343d3c, 0.75, 0.32);
  group.add(box(1.7, 0.12, 1.65, material, x, 0.12, z));
  group.add(box(0.12, 1.05, 1.65, sideMaterial, x - 0.79, 0.65, z));
  group.add(box(0.12, 1.05, 1.65, sideMaterial, x + 0.79, 0.65, z));
  group.add(box(1.7, 1.05, 0.12, sideMaterial, x, 0.65, z - 0.77));
  return group;
}

function positionProduct(group: THREE.Group, progress: number, decision?: string | null): void {
  const normalized = Math.min(Math.max(progress / 100, 0), 1);
  if (normalized <= 0.80) {
    const lineProgress = normalized / 0.80;
    group.position.set(-8.4 + lineProgress * 14.2, 0.84, 0);
    return;
  }
  const sortProgress = (normalized - 0.80) / 0.20;
  const targetZ = decision === "reject" ? 3.1 : decision === "review" ? 0 : -3.1;
  group.position.set(5.8 + sortProgress * 2.4, 0.84 - sortProgress * 0.22, targetZ * sortProgress);
}

export default function TwinScene3D({ snapshot, operatorImageUrl }: TwinScene3DProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<RuntimeScene | null>(null);
  const textureRef = useRef<THREE.Texture | null>(null);
  const liveRef = useRef({
    progress: snapshot.active_cycle?.progress_pct ?? 0,
    phase: snapshot.active_cycle?.phase ?? "idle",
    decision: snapshot.active_cycle?.decision ?? null,
    queueDepth: snapshot.active_cycle?.queue_depth ?? 0,
    active: Boolean(snapshot.active_cycle)
  });

  liveRef.current = {
    progress: snapshot.active_cycle?.progress_pct ?? 0,
    phase: snapshot.active_cycle?.phase ?? "idle",
    decision: snapshot.active_cycle?.decision ?? null,
    queueDepth: snapshot.active_cycle?.queue_depth ?? 0,
    active: Boolean(snapshot.active_cycle)
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111716);
    scene.fog = new THREE.Fog(0x111716, 18, 35);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(11.8, 9.2, 14.5);
    camera.lookAt(0, 0.7, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xc8d8d3, 0x18201e, 1.35));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(4, 12, 8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.left = -16;
    keyLight.shadow.camera.right = 16;
    keyLight.shadow.camera.top = 12;
    keyLight.shadow.camera.bottom = -12;
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x62d9c1, 1.1);
    rimLight.position.set(-8, 5, -8);
    scene.add(rimLight);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(34, 17),
      new THREE.MeshStandardMaterial({ color: 0x202725, metalness: 0.18, roughness: 0.82 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(34, 34, 0x41625a, 0x2b3734);
    grid.position.y = 0.006;
    scene.add(grid);

    const steel = standardMaterial(0x778480, 0.85, 0.25);
    const darkSteel = standardMaterial(0x293230, 0.75, 0.34);
    const black = standardMaterial(0x101514, 0.45, 0.42);
    const accent = standardMaterial(0x1ca77a, 0.45, 0.28);

    scene.add(box(15.8, 0.28, 2.1, darkSteel, -1.0, 0.58, 0));
    scene.add(box(15.6, 0.12, 1.75, black, -1.0, 0.80, 0));
    for (const x of [-7.6, -4.2, -0.8, 2.6, 5.8]) {
      scene.add(box(0.2, 1.2, 1.65, steel, x, 0.02, 0));
    }
    const rollers: THREE.Mesh[] = [];
    for (let x = -8.45; x <= 6.45; x += 0.52) {
      const roller = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 1.64, 18), steel);
      roller.rotation.x = Math.PI / 2;
      roller.position.set(x, 0.84, 0);
      roller.castShadow = true;
      scene.add(roller);
      rollers.push(roller);
    }

    const feeder = new THREE.Group();
    const hopper = new THREE.Mesh(new THREE.CylinderGeometry(1.35, 0.46, 2.2, 4), steel);
    hopper.position.set(-9.5, 2.9, 0);
    hopper.rotation.y = Math.PI / 4;
    hopper.castShadow = true;
    feeder.add(hopper);
    feeder.add(box(0.72, 1.5, 0.72, darkSteel, -9.5, 1.2, 0));
    feeder.add(box(2.1, 0.22, 1.35, steel, -8.55, 0.98, 0));
    scene.add(feeder);

    const cameraGantry = new THREE.Group();
    cameraGantry.add(box(0.22, 3.4, 0.22, steel, -5.75, 2.3, -1.35));
    cameraGantry.add(box(0.22, 3.4, 0.22, steel, -5.75, 2.3, 1.35));
    cameraGantry.add(box(0.45, 0.32, 3.0, steel, -5.75, 3.95, 0));
    const cameraBody = box(0.72, 0.52, 0.72, darkSteel, -5.75, 3.48, 0);
    cameraGantry.add(cameraBody);
    const cameraLensMaterial = new THREE.MeshStandardMaterial({
      color: 0x72d8ff,
      emissive: 0x0b5070,
      emissiveIntensity: 1.2,
      metalness: 0.3,
      roughness: 0.2
    });
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.23, 0.28, 24), cameraLensMaterial);
    lens.position.set(-5.75, 3.12, 0);
    cameraGantry.add(lens);
    const beam = new THREE.Mesh(
      new THREE.ConeGeometry(0.9, 2.3, 32, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x54bfe8, transparent: true, opacity: 0.08, depthWrite: false })
    );
    beam.position.set(-5.75, 1.9, 0);
    beam.rotation.z = Math.PI;
    cameraGantry.add(beam);
    scene.add(cameraGantry);
    const cameraLight = new THREE.PointLight(0x56c9ff, 0.5, 5);
    cameraLight.position.set(-5.75, 2.7, 0);
    scene.add(cameraLight);

    const inspectionMaterial = standardMaterial(0x52605c, 0.82, 0.24);
    const inspection = new THREE.Group();
    inspection.add(box(2.25, 3.0, 3.25, inspectionMaterial, -2.3, 2.1, 0));
    inspection.add(box(1.4, 1.25, 3.4, black, -2.3, 1.25, 0));
    for (const z of [-1.25, 1.25]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.07, 12, 32), accent);
      ring.rotation.y = Math.PI / 2;
      ring.position.set(-2.3, 2.25, z);
      inspection.add(ring);
    }
    const scanPlane = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 1.1, 1.65),
      new THREE.MeshBasicMaterial({ color: 0x5fffd1, transparent: true, opacity: 0.32 })
    );
    scanPlane.position.set(-2.3, 1.35, 0);
    inspection.add(scanPlane);
    scene.add(inspection);

    const aiCabinet = new THREE.Group();
    aiCabinet.add(box(1.5, 2.9, 1.25, darkSteel, 1.2, 1.55, -2.6));
    aiCabinet.add(box(1.12, 0.78, 0.08, black, 1.2, 2.05, -1.94));
    for (let index = 0; index < 4; index += 1) {
      const led = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 12, 8),
        new THREE.MeshBasicMaterial({ color: index === 0 ? 0x3dff9a : 0x6f8d85 })
      );
      led.position.set(0.84 + index * 0.24, 1.3, -1.92);
      aiCabinet.add(led);
    }
    scene.add(aiCabinet);

    const monitorMaterial = new THREE.MeshBasicMaterial({ color: 0x0d1815 });
    const monitorFrame = new THREE.Group();
    monitorFrame.add(box(2.5, 1.65, 0.16, steel, 3.2, 3.1, -3.0));
    const monitorScreen = new THREE.Mesh(new THREE.PlaneGeometry(2.15, 1.3), monitorMaterial);
    monitorScreen.position.set(3.2, 3.1, -2.91);
    monitorFrame.add(monitorScreen);
    monitorFrame.add(box(0.18, 1.35, 0.18, steel, 3.2, 1.62, -3.0));
    monitorFrame.add(box(1.4, 0.14, 0.65, steel, 3.2, 0.94, -3.0));
    scene.add(monitorFrame);

    const gatePivot = new THREE.Group();
    gatePivot.position.set(5.65, 1.28, 0);
    gatePivot.add(box(0.18, 1.75, 0.18, steel, 0, 0, 0));
    const gateArm = box(0.2, 0.16, 2.25, standardMaterial(0xe1ad38, 0.45, 0.28), 0, 0.4, 1.05);
    gatePivot.add(gateArm);
    scene.add(gatePivot);

    scene.add(createBin(0x1b8f62, 8.6, -3.1));
    scene.add(createBin(0xc28a27, 8.6, 0));
    scene.add(createBin(0xb94334, 8.6, 3.1));

    const activeProduct = createProduct();
    activeProduct.visible = false;
    scene.add(activeProduct);
    const queueMaterial = standardMaterial(0x4caaa0, 0.48, 0.32);
    const queuedProducts: THREE.Mesh[] = [];
    for (let index = 0; index < 6; index += 1) {
      const queued = box(0.72, 0.15, 0.55, queueMaterial, -9.7 - index * 0.42, 0.92, -0.95);
      queued.visible = false;
      scene.add(queued);
      queuedProducts.push(queued);
    }

    const decisionHaloMaterial = new THREE.MeshBasicMaterial({
      color: 0x42d890,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide
    });
    const decisionHalo = new THREE.Mesh(new THREE.RingGeometry(0.48, 0.62, 32), decisionHaloMaterial);
    decisionHalo.rotation.x = -Math.PI / 2;
    decisionHalo.position.y = 0.73;
    decisionHalo.visible = false;
    activeProduct.add(decisionHalo);

    const observer = new ResizeObserver(() => {
      const width = Math.max(mount.clientWidth, 1);
      const height = Math.max(mount.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    });
    observer.observe(mount);

    const clock = new THREE.Clock();
    const animate = () => {
      const elapsed = clock.getElapsedTime();
      const live = liveRef.current;
      activeProduct.visible = live.active;
      if (live.active) positionProduct(activeProduct, live.progress, live.decision);

      queuedProducts.forEach((queued, index) => {
        queued.visible = index < Math.min(live.queueDepth, queuedProducts.length);
        queued.position.x = -9.7 - index * 0.42;
        queued.position.z = -0.95 + (index % 2) * 0.5;
      });
      rollers.forEach((roller) => {
        roller.rotation.z = elapsed * 2.4;
      });

      const captureActive = live.phase === "capture";
      const inspectionActive = live.phase === "inspection";
      const warning = Boolean(live.decision && live.decision !== "pass");
      cameraLight.intensity = captureActive ? 5.5 + Math.sin(elapsed * 24) * 1.5 : warning ? 2.2 : 0.55;
      cameraLight.color.setHex(warning ? 0xff5747 : 0x56c9ff);
      cameraLensMaterial.emissive.setHex(warning ? 0x9c140e : 0x0b5070);
      cameraLensMaterial.emissiveIntensity = captureActive ? 4 : warning ? 3 : 1.2;

      inspectionMaterial.emissive.setHex(warning ? 0x7a120d : inspectionActive ? 0x064d39 : 0x000000);
      inspectionMaterial.emissiveIntensity = warning ? 1.35 : inspectionActive ? 0.8 : 0;
      scanPlane.visible = live.phase === "inspection" || live.phase === "decision";
      scanPlane.position.y = 1.25 + Math.sin(elapsed * 7) * 0.48;

      const gateTarget = live.phase === "sorting" && live.decision === "reject" ? -0.82 : 0;
      gatePivot.rotation.y += (gateTarget - gatePivot.rotation.y) * 0.14;

      decisionHalo.visible = Boolean(live.decision);
      if (live.decision) {
        decisionHaloMaterial.color.setHex(
          live.decision === "reject" ? 0xff5544 : live.decision === "review" ? 0xffb43b : 0x42d890
        );
        decisionHalo.scale.setScalar(1 + Math.sin(elapsed * 7) * 0.08);
      }

      renderer.render(scene, camera);
      if (runtimeRef.current) runtimeRef.current.frameId = requestAnimationFrame(animate);
    };

    runtimeRef.current = {
      renderer,
      scene,
      camera,
      activeProduct,
      queuedProducts,
      rollers,
      gatePivot,
      scanPlane,
      cameraLight,
      cameraLensMaterial,
      inspectionMaterial,
      decisionHalo,
      decisionHaloMaterial,
      monitorMaterial,
      frameId: requestAnimationFrame(animate),
      observer
    };

    return () => {
      const runtime = runtimeRef.current;
      if (!runtime) return;
      cancelAnimationFrame(runtime.frameId);
      runtime.observer.disconnect();
      runtime.scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      });
      textureRef.current?.dispose();
      runtime.renderer.dispose();
      runtime.renderer.domElement.remove();
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const material = runtimeRef.current?.monitorMaterial;
    if (!material || !operatorImageUrl) return;
    let cancelled = false;
    const loader = new THREE.TextureLoader();
    loader.load(mediaUrl(operatorImageUrl), (texture) => {
      if (cancelled) {
        texture.dispose();
        return;
      }
      texture.colorSpace = THREE.SRGBColorSpace;
      textureRef.current?.dispose();
      textureRef.current = texture;
      material.map = texture;
      material.color.setHex(0xffffff);
      material.needsUpdate = true;
    });
    return () => {
      cancelled = true;
    };
  }, [operatorImageUrl]);

  return <div ref={mountRef} className="ops-twin-canvas" aria-label="Live 3D inspection line" />;
}
