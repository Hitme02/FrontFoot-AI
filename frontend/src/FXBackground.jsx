// src/FXBackground.jsx
import * as React from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { PerformanceMonitor } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";

/** Wavy plane */
function WavePlane({ paused }) {
  const mat = React.useRef();
  const t = React.useRef(0);
  useFrame((_, d) => {
    if (paused) return;
    t.current += d;
    if (mat.current) mat.current.uniforms.uTime.value = t.current;
  });

  const uniforms = React.useMemo(() => ({
    uTime: { value: 0 },
    uColorA: { value: new THREE.Color("#2af598") },
    uColorB: { value: new THREE.Color("#ff6ec7") },
    uColorC: { value: new THREE.Color("#009EFD") },
  }), []);

  return (
    <mesh position={[0, 0, -6]} rotation={[-0.2, 0, 0]}>
      <planeGeometry args={[16, 10, 120, 90]} />
      <shaderMaterial
        ref={mat}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        uniforms={uniforms}
        vertexShader={/* glsl */`
          varying vec2 vUv; uniform float uTime;
          void main(){
            vUv = uv;
            vec3 p = position;
            p.z += sin(p.x*0.6 + uTime*0.8)*0.28;
            p.z += cos(p.y*0.7 + uTime*0.6)*0.22;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(p,1.0);
          }`}
        fragmentShader={/* glsl */`
          varying vec2 vUv;
          uniform vec3 uColorA,uColorB,uColorC;
          void main(){
            float b1 = smoothstep(0.15,0.35,vUv.y)-smoothstep(0.35,0.55,vUv.y);
            float b2 = smoothstep(0.45,0.65,vUv.x)-smoothstep(0.65,0.85,vUv.x);
            vec3 col = mix(uColorA,uColorB,vUv.x);
            col = mix(col,uColorC,b2*0.8);
            col += b1*0.5;
            gl_FragColor = vec4(col,0.32);
          }`}
      />
    </mesh>
  );
}

/** Particles */
function NeonParticles({ count = 600, paused }) {
  const geom = React.useRef();
  const vel = React.useRef(new Float32Array(count));

  const positions = React.useMemo(() => {
    const arr = new Float32Array(count*3);
    for (let i=0;i<count;i++){
      arr[i*3+0]=(Math.random()-0.5)*16;
      arr[i*3+1]=(Math.random()-0.5)*10;
      arr[i*3+2]=Math.random()*-6-1;
      vel.current[i]=0.4+Math.random()*0.8;
    }
    return arr;
  }, [count]);

  React.useEffect(()=>{
    geom.current.setAttribute("position", new (THREE.BufferAttribute)(positions,3));
  },[positions]);

  useFrame((_,d)=>{
    if (paused) return;
    const pos=geom.current.attributes.position.array;
    for(let i=0;i<count;i++){
      const iy=i*3+1;
      pos[iy]+=vel.current[i]*d*0.3;
      if(pos[iy]>6) pos[iy]=-6;
    }
    geom.current.attributes.position.needsUpdate=true;
  });

  return (
    <points>
      <bufferGeometry ref={geom} />
      <pointsMaterial
        size={0.035}
        sizeAttenuation
        transparent
        depthWrite={false}
        color={new THREE.Color("#bbffe9")}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export default function FXBackground({ enabled = true, paused = false }) {
  const [dpr, setDpr] = React.useState(1.5);
  const [low, setLow] = React.useState(false);

  if (!enabled) return null;

  return (
    <div className="webgl-bg">
      <Canvas
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
        dpr={dpr}
        camera={{ position: [0, 0, 5.5], fov: 60 }}
      >
        <PerformanceMonitor
          onDecline={() => { setDpr(1); setLow(true); }}
          onIncline={() => { setDpr(1.5); setLow(false); }}
        />
        <ambientLight intensity={0.5} />
        <WavePlane paused={paused} />
        <NeonParticles count={low ? 350 : 700} paused={paused} />
        {!low && !paused && (
          <EffectComposer multisampling={0}>
            <Bloom intensity={0.65} luminanceThreshold={0.12} luminanceSmoothing={0.3} />
          </EffectComposer>
        )}
      </Canvas>
    </div>
  );
}
