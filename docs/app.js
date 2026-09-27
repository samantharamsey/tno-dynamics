import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const mSun = 1.9890e30;
const mNeptune = 1.0241e26;
const mu = mNeptune/(mSun + mNeptune);
const cMin = 2.99;
const cMax = 3.02;
const sliderScale = 3;

function newton(fn, guess) {
  let g = guess;
  for (let n = 0; n < 50; n++) {
    const [f, df] = fn(g);
    const next = g - f/df;
    if (Math.abs(next - g) < 1e-14) return next;
    g = next;
  }
  return g;
}

const g1 = newton(g => [
  g**5 - (3 - mu)*g**4 + (3 - 2*mu)*g**3 - mu*g**2 + 2*mu*g - mu,
  5*g**4 - 4*(3 - mu)*g**3 + 3*(3 - 2*mu)*g**2 - 2*mu*g + 2*mu
], 0.1);
const g2 = newton(g => [
  g**5 + (3 - mu)*g**4 + (3 - 2*mu)*g**3 - mu*g**2 - 2*mu*g - mu,
  5*g**4 + 4*(3 - mu)*g**3 + 3*(3 - 2*mu)*g**2 - 2*mu*g - 2*mu
], 0.1);
const l1 = 1 - mu - g1;
const l2 = 1 - mu + g2;

function potential(x, y, z) {
  const d = Math.hypot(x + mu, y, z);
  const r = Math.hypot(x - 1 + mu, y, z);
  return (1 - mu)/d + mu/r + 0.5*(x*x + y*y);
}

const c1 = 2*potential(l1, 0, 0);
const c2 = 2*potential(l2, 0, 0);
const cCenter = (c1 + c2)/2;

function sliderToC(s) {
  const f = (10**(sliderScale*Math.abs(s)) - 1)/(10**sliderScale - 1);
  return s < 0
    ? cCenter - (cCenter - cMin)*f
    : cCenter + (cMax - cCenter)*f;
}

const vertexShader = `
  varying vec3 vWorldPosition;
  void main() {
    vec4 world = modelMatrix*vec4(position, 1.0);
    vWorldPosition = world.xyz;
    gl_Position = projectionMatrix*viewMatrix*world;
  }
`;

const fragmentShader = `
  precision highp float;
  varying vec3 vWorldPosition;
  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;
  uniform float uC;
  uniform float uMu;
  uniform int uSteps;

  float field(vec3 p) {
    float d = max(length(p - vec3(-uMu, 0.0, 0.0)), 0.000015);
    float r = max(length(p - vec3(1.0 - uMu, 0.0, 0.0)), 0.000015);
    float U = (1.0 - uMu)/d + uMu/r + 0.5*(p.x*p.x + p.y*p.y);
    return 2.0*U - uC;
  }

  vec2 hitBox(vec3 ro, vec3 rd) {
    vec3 inv = 1.0/rd;
    vec3 lo = (uBoundsMin - ro)*inv;
    vec3 hi = (uBoundsMax - ro)*inv;
    vec3 near3 = min(lo, hi);
    vec3 far3 = max(lo, hi);
    return vec2(max(max(near3.x, near3.y), near3.z),
                min(min(far3.x, far3.y), far3.z));
  }

  vec3 normalAt(vec3 p) {
    float e = max(length(uBoundsMax - uBoundsMin)*0.00012, 0.000002);
    return normalize(vec3(
      field(p + vec3(e,0,0)) - field(p - vec3(e,0,0)),
      field(p + vec3(0,e,0)) - field(p - vec3(0,e,0)),
      field(p + vec3(0,0,e)) - field(p - vec3(0,0,e))
    ));
  }

  void main() {
    vec3 ro = cameraPosition;
    vec3 rd = normalize(vWorldPosition - ro);
    vec2 hit = hitBox(ro, rd);
    float start = max(hit.x, 0.0);
    float finish = hit.y;
    if (finish <= start) discard;

    float previousT = start;
    float previousF = field(ro + rd*previousT);
    vec3 accumulatedColor = vec3(0.0);
    float accumulatedAlpha = 0.0;
    int surfaceCount = 0;
    for (int n = 1; n <= 192; n++) {
      if (n > uSteps) break;
      float t = mix(start, finish, float(n)/float(uSteps));
      float currentF = field(ro + rd*t);
      if ((previousF <= 0.0 && currentF >= 0.0) ||
          (previousF >= 0.0 && currentF <= 0.0)) {
        float a = previousT;
        float b = t;
        float fa = previousF;
        for (int j = 0; j < 7; j++) {
          float middle = 0.5*(a + b);
          float fm = field(ro + rd*middle);
          if ((fa <= 0.0 && fm <= 0.0) || (fa >= 0.0 && fm >= 0.0)) {
            a = middle;
            fa = fm;
          } else {
            b = middle;
          }
        }
        vec3 p = ro + rd*(0.5*(a + b));
        vec3 normal = normalAt(p);
        vec3 light = normalize(vec3(0.5, -0.3, 1.0));
        float diffuse = 0.35 + 0.65*abs(dot(normal, light));
        vec3 edge = min(p - uBoundsMin, uBoundsMax - p);
        float edgeDistance = min(edge.x, min(edge.y, edge.z));
        float fadeWidth = length(uBoundsMax - uBoundsMin)*0.018;
        float layerAlpha = 0.34*smoothstep(0.0, fadeWidth, edgeDistance);
        accumulatedColor += (1.0 - accumulatedAlpha)*layerAlpha*uColor*diffuse;
        accumulatedAlpha += (1.0 - accumulatedAlpha)*layerAlpha;
        surfaceCount++;
        if (surfaceCount >= 4) break;
      }
      previousT = t;
      previousF = currentF;
    }
    if (accumulatedAlpha <= 0.0) discard;
    gl_FragColor = vec4(accumulatedColor/accumulatedAlpha, accumulatedAlpha);
  }
`;

function labelSprite(text, color = '#ffffff', scale = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  context.font = '30px Arial';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(text, 128, 32);
  const material = new THREE.SpriteMaterial({map: new THREE.CanvasTexture(canvas), depthTest: false});
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.24*scale, 0.06*scale, 1);
  sprite.renderOrder = 4;
  return sprite;
}

function marker(radius, color) {
  const object = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 14),
    new THREE.MeshStandardMaterial({color, roughness: 0.65, metalness: 0.05, depthTest: false})
  );
  object.renderOrder = 3;
  return object;
}

function addMarker(scene, position, radius, color, label, labelScale) {
  const object = marker(radius, color);
  object.position.copy(position);
  scene.add(object);
  const sprite = labelSprite(label, '#ffffff', labelScale);
  sprite.position.copy(position).add(new THREE.Vector3(0, radius*2.2, 0));
  scene.add(sprite);
}

function createView(element, bounds, steps, closeup = false) {
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor(0x000000, 1);
  element.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(2, -1, 3);
  scene.add(light);

  const min = new THREE.Vector3(...bounds[0]);
  const max = new THREE.Vector3(...bounds[1]);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  const span = Math.max(size.x, size.y, size.z);

  const camera = new THREE.PerspectiveCamera(38, 1, span*0.0001, span*20);
  camera.position.copy(center).add(new THREE.Vector3(span*1.35, -span*1.55, span*0.9));
  camera.lookAt(center);

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uBoundsMin: {value: min},
      uBoundsMax: {value: max},
      uColor: {value: new THREE.Color('#d8b4fe')},
      uC: {value: cCenter},
      uMu: {value: mu},
      uSteps: {value: steps}
    }
  });
  const box = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), material);
  box.position.copy(center);
  scene.add(box);

  if (closeup) {
    addMarker(scene, new THREE.Vector3(1 - mu, 0, 0), 0.002, 0x4f8cff, 'neptune', 0.42);
    addMarker(scene, new THREE.Vector3(l1, 0, 0), 0.0012, 0xffd700, 'L1', 0.34);
    addMarker(scene, new THREE.Vector3(l2, 0, 0), 0.0012, 0xffd700, 'L2', 0.34);
  } else {
    addMarker(scene, new THREE.Vector3(-mu, 0, 0), 0.035, 0xffd166, 'sun', 1.0);
    addMarker(scene, new THREE.Vector3(1 - mu, 0, 0), 0.022, 0x4f8cff, 'neptune', 1.0);
    const points = [];
    for (let n = 0; n <= 180; n++) {
      const angle = 2*Math.PI*n/180;
      points.push(new THREE.Vector3((1 - mu)*Math.cos(angle), (1 - mu)*Math.sin(angle), 0));
    }
    const orbit = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineDashedMaterial({color: 0x888888, dashSize: 0.025, gapSize: 0.018, transparent: true, opacity: 0.55})
    );
    orbit.computeLineDistances();
    scene.add(orbit);
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.enableDamping = false;
  controls.update();

  function render() {
    const width = element.clientWidth;
    const height = element.clientHeight;
    if (renderer.domElement.width !== Math.round(width*renderer.getPixelRatio()) ||
        renderer.domElement.height !== Math.round(height*renderer.getPixelRatio())) {
      renderer.setSize(width, height, false);
      camera.aspect = width/height;
      camera.updateProjectionMatrix();
    }
    renderer.render(scene, camera);
  }
  controls.addEventListener('change', render);
  new ResizeObserver(render).observe(element);
  return {material, render};
}

const system = createView(
  document.getElementById('system-view'),
  [[-1.5, -1.5, -1], [1.5, 1.5, 1]],
  176
);
const neptune = createView(
  document.getElementById('neptune-view'),
  [[l1 - 0.08, -0.16, -0.16], [l2 + 0.08, 0.16, 0.16]],
  160,
  true
);

const slider = document.getElementById('c-slider');
const readout = document.getElementById('readout');
function update() {
  const C = sliderToC(Number(slider.value));
  readout.textContent = `C = ${C.toFixed(8)}`;
  system.material.uniforms.uC.value = C;
  neptune.material.uniforms.uC.value = C;
  system.render();
  neptune.render();
}
slider.addEventListener('input', update);
update();
