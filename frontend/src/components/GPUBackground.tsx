import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export function GPUBackground() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: true 
    });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);

    // Create RTX 4090-style GPU
    const createRTX4090 = () => {
      const group = new THREE.Group();

      // Main PCB (green circuit board)
      const pcbGeometry = new THREE.BoxGeometry(8, 0.15, 4);
      const pcbMaterial = new THREE.MeshStandardMaterial({
        color: 0x0a4d0a,
        metalness: 0.3,
        roughness: 0.7,
      });
      const pcb = new THREE.Mesh(pcbGeometry, pcbMaterial);
      pcb.position.y = -0.5;
      group.add(pcb);

      // GPU Die (center chip - silver/gray)
      const dieGeometry = new THREE.BoxGeometry(2.5, 0.4, 2);
      const dieMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a,
        metalness: 0.9,
        roughness: 0.1,
        emissive: 0x111111,
      });
      const die = new THREE.Mesh(dieGeometry, dieMaterial);
      die.position.y = -0.25;
      group.add(die);

      // NVIDIA logo area (green accent on chip)
      const logoGeometry = new THREE.BoxGeometry(1.5, 0.05, 0.5);
      const logoMaterial = new THREE.MeshStandardMaterial({
        color: 0x76b900,
        metalness: 0.8,
        roughness: 0.2,
        emissive: 0x76b900,
        emissiveIntensity: 0.4,
      });
      const logo = new THREE.Mesh(logoGeometry, logoMaterial);
      logo.position.set(0, 0.2, -0.5);
      group.add(logo);

      // Memory chips (8 black rectangles around the die)
      const memoryGeometry = new THREE.BoxGeometry(0.6, 0.25, 0.8);
      const memoryMaterial = new THREE.MeshStandardMaterial({
        color: 0x0d0d0d,
        metalness: 0.6,
        roughness: 0.3,
      });

      const memoryPositions = [
        [-3, -0.3, 1.2], [3, -0.3, 1.2],
        [-3, -0.3, -1.2], [3, -0.3, -1.2],
        [-2, -0.3, 1.8], [2, -0.3, 1.8],
        [-2, -0.3, -1.8], [2, -0.3, -1.8],
      ];

      memoryPositions.forEach(pos => {
        const memory = new THREE.Mesh(memoryGeometry, memoryMaterial);
        memory.position.set(pos[0], pos[1], pos[2]);
        group.add(memory);
      });

      // Heatsink/Shroud (top cover - black with RGB)
      const shroudGeometry = new THREE.BoxGeometry(8.5, 1, 4.5);
      const shroudMaterial = new THREE.MeshStandardMaterial({
        color: 0x0a0a0a,
        metalness: 0.8,
        roughness: 0.2,
        emissive: 0x050505,
      });
      const shroud = new THREE.Mesh(shroudGeometry, shroudMaterial);
      shroud.position.y = 0.5;
      group.add(shroud);

      // RGB light strips (colorful glow on edges)
      const rgbGeometry = new THREE.BoxGeometry(8.3, 0.1, 0.2);
      const rgbMaterial1 = new THREE.MeshStandardMaterial({
        color: 0x00ff88,
        metalness: 0.9,
        roughness: 0.1,
        emissive: 0x00ff88,
        emissiveIntensity: 0.8,
      });
      const rgbStrip1 = new THREE.Mesh(rgbGeometry, rgbMaterial1);
      rgbStrip1.position.set(0, 0.5, 2.2);
      group.add(rgbStrip1);

      const rgbMaterial2 = new THREE.MeshStandardMaterial({
        color: 0x0088ff,
        metalness: 0.9,
        roughness: 0.1,
        emissive: 0x0088ff,
        emissiveIntensity: 0.8,
      });
      const rgbStrip2 = new THREE.Mesh(rgbGeometry, rgbMaterial2);
      rgbStrip2.position.set(0, 0.5, -2.2);
      group.add(rgbStrip2);

      // Cooling fans (3 fans)
      const createFan = (x: number, z: number) => {
        const fanGroup = new THREE.Group();
        
        // Fan housing
        const housingGeometry = new THREE.CylinderGeometry(0.8, 0.8, 0.2, 32);
        const housingMaterial = new THREE.MeshStandardMaterial({
          color: 0x1a1a1a,
          metalness: 0.7,
          roughness: 0.3,
        });
        const housing = new THREE.Mesh(housingGeometry, housingMaterial);
        housing.rotation.x = Math.PI / 2;
        fanGroup.add(housing);

        // Fan blades
        for (let i = 0; i < 9; i++) {
          const bladeGeometry = new THREE.BoxGeometry(0.15, 0.05, 0.7);
          const bladeMaterial = new THREE.MeshStandardMaterial({
            color: 0x2a2a2a,
            metalness: 0.8,
            roughness: 0.2,
          });
          const blade = new THREE.Mesh(bladeGeometry, bladeMaterial);
          blade.position.y = 0.05;
          blade.rotation.y = (i * Math.PI * 2) / 9;
          fanGroup.add(blade);
        }

        fanGroup.position.set(x, 1.1, z);
        return fanGroup;
      };

      const fan1 = createFan(-2.5, 0);
      const fan2 = createFan(0, 0);
      const fan3 = createFan(2.5, 0);
      group.add(fan1, fan2, fan3);

      // Power connectors (12VHPWR on the side)
      const connectorGeometry = new THREE.BoxGeometry(0.3, 0.5, 0.8);
      const connectorMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a,
        metalness: 0.6,
        roughness: 0.4,
      });
      const connector = new THREE.Mesh(connectorGeometry, connectorMaterial);
      connector.position.set(4.2, 0.3, -1.5);
      group.add(connector);

      // Golden pins on connector
      const pinGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.3, 8);
      const pinMaterial = new THREE.MeshStandardMaterial({
        color: 0xffd700,
        metalness: 1,
        roughness: 0.1,
      });
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 3; j++) {
          const pin = new THREE.Mesh(pinGeometry, pinMaterial);
          pin.position.set(4.35, 0.2 + i * 0.15, -1.7 + j * 0.2);
          pin.rotation.z = Math.PI / 2;
          group.add(pin);
        }
      }

      // Edge highlights
      const edges = new THREE.EdgesGeometry(shroudGeometry);
      const edgeMaterial = new THREE.LineBasicMaterial({ 
        color: 0x333333,
        linewidth: 1
      });
      const edgeLines = new THREE.LineSegments(edges, edgeMaterial);
      edgeLines.position.copy(shroud.position);
      group.add(edgeLines);

      // Store fans for animation
      (group as any).fans = [fan1, fan2, fan3];
      (group as any).rgbStrips = [rgbStrip1, rgbStrip2];

      return group;
    };

    const gpu = createRTX4090();
    scene.add(gpu);

    // Lighting (dramatic GPU showcase lighting)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(5, 8, 5);
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
    fillLight.position.set(-5, 3, -5);
    scene.add(fillLight);

    const accentLight = new THREE.PointLight(0x76b900, 1, 20);
    accentLight.position.set(0, 2, 0);
    scene.add(accentLight);

    // Camera position (showcase angle)
    camera.position.set(6, 4, 8);
    camera.lookAt(0, 0, 0);

    // Animation
    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.01;

      // Gentle GPU rotation
      gpu.rotation.y = Math.sin(time * 0.2) * 0.2 + 0.3;
      gpu.rotation.x = Math.cos(time * 0.15) * 0.1;

      // Spin the cooling fans
      if ((gpu as any).fans) {
        (gpu as any).fans.forEach((fan: THREE.Group, index: number) => {
          fan.rotation.z = time * 3 + index * 0.5;
        });
      }

      // Pulse RGB strips
      if ((gpu as any).rgbStrips) {
        const pulse = Math.sin(time * 2) * 0.3 + 0.5;
        (gpu as any).rgbStrips.forEach((strip: THREE.Mesh) => {
          (strip.material as THREE.MeshStandardMaterial).emissiveIntensity = pulse;
        });
      }

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 50,
        pointerEvents: 'none',
        opacity: 1,
      }}
    />
  );
}
