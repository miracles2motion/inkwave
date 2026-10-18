// Renderer + post stack (MSAA HDR target → optional GTAO → bloom → grade/vignette → output).
// Apple GPUs (ANGLE's Metal backend): multisampled half-float targets cost ~30 ms/frame at 1920×1200 on an M1 Pro,
// so there the HDR target is single-sampled and edges are smoothed by SMAA after output, and GTAO runs at half res.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';
import { QUALITY, rendererQuality } from '../config.js';
import { G } from './ctx.js';
import { shouldUseTouch, isMobileOrTouch } from './touch.js';

const BLOOM = [0.28, 0.45, 2.4];   // default bloom: strength, radius, HDR threshold

const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    uSat: { value: 1.08 },
    uVib: { value: 0.12 },                               // extra saturation for muted colours only (ink never clips)
    uContrast: { value: 1.07 },                          // log-space contrast around mid grey
    uShadowTint: { value: new THREE.Vector3(0.975, 0.99, 1.035) },
    uHighTint: { value: new THREE.Vector3(1.025, 1.0, 0.972) },
    uLift: { value: 0.0 },
    uExposure: { value: 1.0 },                           // per-theme exposure (the renderer's own exposure stays fixed: the lobby set shares it)
    uVignette: { value: 0.22 },
    uHurt: { value: 0 },
    uHurtColor: { value: new THREE.Color(1, 0.2, 0.3) },
    uFlash: { value: 0 },
    uAspect: { value: 1.7 },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform float uSat; uniform float uVignette; uniform float uHurt; uniform vec3 uHurtColor; uniform float uFlash; uniform float uAspect;
    uniform float uVib; uniform float uContrast; uniform vec3 uShadowTint; uniform vec3 uHighTint; uniform float uLift; uniform float uExposure;
    varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      c.rgb *= uExposure;
      float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      // vibrance: muted colours gain saturation, already-saturated ones (team ink) barely move
      float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b));
      float chroma = (mx - mn) / max(mx, 1e-4);
      c.rgb = max(mix(vec3(l), c.rgb, uSat + uVib * (1.0 - smoothstep(0.1, 0.7, chroma))), 0.0);
      // contrast in log space around mid grey (keeps HDR highlights ordered), then a cool-shadow / warm-light split tone
      c.rgb = 0.18 * pow(max(c.rgb, vec3(1e-6)) / 0.18, vec3(uContrast)) + uLift;
      // split tone is for the world's neutrals: strongly saturated colours (team ink) keep their exact hue
      float lt = smoothstep(0.015, 0.55, l);
      c.rgb *= mix(vec3(1.0), mix(uShadowTint, uHighTint, lt), 1.0 - 0.85 * smoothstep(0.35, 0.8, chroma));
      vec2 q = (vUv - 0.5) * vec2(uAspect, 1.0);
      float r = length(q);
      float v = smoothstep(0.55, 1.25, r);
      c.rgb *= 1.0 - uVignette * v;
      // low health: the HUD draws the coloured edge; here we only drain saturation + darken the rim slightly
      float lum = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      c.rgb = mix(c.rgb, vec3(lum), uHurt * 0.45);
      c.rgb *= 1.0 - uHurt * 0.25 * smoothstep(0.4, 1.2, r);
      c.rgb += uFlash;
      gl_FragColor = c;
    }`,
};

// r186's PCF filter uses a 5-tap rotated Vogel disk with per-pixel noise, which reads as grainy stipple on every soft
// shadow edge. Swap it for a noise-free 3×3 grid of hardware-compared (bilinear) taps: smooth and temporally stable.
(function patchShadowFilter() {
  const chunk = THREE.ShaderChunk.shadowmap_pars_fragment;
  const re = /shadow = \(\s*texture\( shadowMap, vec3\( shadowCoord\.xy \+ vogelDiskSample\( 0, 5, phi \) \* radius, shadowCoord\.z \) \)[\s\S]*?\) \* 0\.2;/;
  if (!re.test(chunk)) { console.warn('[inkwave] shadow chunk layout changed; keeping stock PCF'); return; }
  THREE.ShaderChunk.shadowmap_pars_fragment = chunk.replace(re, `vec2 ts = texelSize * max( shadowRadius * 0.55, 0.6 );
				float s9 = 0.0;
				for ( int sx = -1; sx <= 1; sx ++ ) for ( int sy = -1; sy <= 1; sy ++ ) s9 += texture( shadowMap, vec3( shadowCoord.xy + vec2( float( sx ), float( sy ) ) * ts, shadowCoord.z ) );
				shadow = s9 * ( 1.0 / 9.0 );`);
})();

function isAppleGPU(gl) {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const name = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  return /Apple/i.test(name) && !/Intel|AMD|Radeon|NVIDIA/i.test(name);
}

export class Renderer {
  constructor(container, settings) {
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', stencil: false }));
    // Shader status checks make every compile synchronous (the main thread waits on the GPU process for the link
    // result): a mid-game compile then stalls input, netcode and all. Players run without them; ?shadercheck (the
    // smoke test and dev tools) turns them back on so a broken shader still fails loudly.
    r.debug.checkShaderErrors = typeof location !== 'undefined' && new URLSearchParams(location.search).has('shadercheck');
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.NeutralToneMapping;
    r.toneMappingExposure = 1.0;
    r.info.autoReset = false;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    r.setClearColor(0x9fd8f0, 1);
    container.appendChild(r.domElement);
    this.appleGPU = isAppleGPU(r.getContext());
    r.domElement.id = 'game-canvas';
    this.container = container;
    this.scene = null; this.camera = null;
    this.settings = settings;
    this.touch = shouldUseTouch(settings);
    this.q = rendererQuality(settings, this.touch);
    this._w = 0; this._h = 0;
  }

  setScene(scene, camera) {
    this.scene = scene; this.camera = camera;
    this._buildComposer();
  }

  _buildComposer() {
    const r = this.renderer, q = this.q;
    if (this.composer) { this.composer.renderTarget1.dispose(); this.composer.renderTarget2.dispose(); }
    this.dynScale = this.dynScale || 1;
    const pr = Math.min(window.devicePixelRatio || 1, q.pixelRatio) * this.dynScale;
    r.setPixelRatio(pr);
    const el = this.container || (typeof document !== 'undefined' ? document.getElementById('app') : null) || (typeof document !== 'undefined' ? document.body : null);
    const w = Math.round(el?.clientWidth || window.innerWidth || 1);
    const h = Math.round(el?.clientHeight || window.innerHeight || 1);
    r.setSize(w, h, false);
    if (r.domElement) {
      r.domElement.style.width = '100%';
      r.domElement.style.height = '100%';
    }
    // Direct render fast-path: bypass EffectComposer completely on low quality and mobile medium without bloom/AO
    this.useDirectRender = !!(
      q.directRender ||
      this.settings?.quality === 'low' ||
      (this.touch && this.settings?.quality !== 'ultra' && !q.bloom && !q.ao) ||
      (isMobileOrTouch() && this.settings?.quality !== 'ultra' && !q.bloom && !q.ao)
    );
    // MSAA is skipped where it costs the most. On Apple GPUs a multisampled half-float target is brutally slow; on a
    // touch device it is the most bandwidth-hungry object in the frame and a phone has the least bandwidth to give.
    // The trade is worth making because MSAA charges every frame for an edge-smoothing benefit that a single
    // full-screen SMAA pass (added below) delivers for a fraction of the cost.
    this.noMSAA = this.appleGPU || this.touch;
    this.samples = this.noMSAA ? 0 : q.msaa || 0;
    const rt = new THREE.WebGLRenderTarget(w * pr, h * pr, { type: THREE.HalfFloatType, samples: this.samples });
    const comp = (this.composer = new EffectComposer(r, rt));
    comp.setPixelRatio(pr);
    comp.setSize(w, h);
    this.renderPass = new RenderPass(this.scene, this.camera);
    comp.addPass(this.renderPass);
    this.gtao = null;
    if (q.ao) {
      const ao = (this.gtao = new GTAOPass(this.scene, this.camera, w, h));
      ao.output = GTAOPass.OUTPUT.Default;
      ao.blendIntensity = 0.8;   // never crush to black: fully enclosed spots (tunnel mouths, under the overpass) keep 20 % of their light
      // contact AO: a metre-scale radius so crates, planters and pillar feet sit on the deck (it multiplies the lit
      // frame, so it stays short-range — no dirty halos on sunlit walls)
      ao.updateGtaoMaterial({ radius: 1.1, distanceExponent: 1.6, thickness: 1.0, scale: 1.5, samples: 12, distanceFallOff: 1.0 });
      // Half-resolution AO with bilateral filter: delivers smooth soft ambient contact shadows
      // while cutting 75% of GPU raymarching overhead to keep devices cool.
      if (this.appleGPU || this.settings?.quality !== 'ultra') {
        const set = ao.setSize.bind(ao);
        ao.setSize = (sw, sh) => set(Math.ceil(sw / 2), Math.ceil(sh / 2));
      }
      comp.addPass(ao);
    }
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), ...BLOOM);
    this.bloom.enabled = !!(q.bloom && this.settings.bloom);
    comp.addPass(this.bloom);
    this.grade = new ShaderPass(GradeShader);
    this._gradeSrc = null;   // (re)apply the theme grade + bloom to the new passes
    comp.addPass(this.grade);
    // optional screen-FX pass (src/fx/screenfx.js) — runs in HDR linear space before tone mapping/output
    if (this.extraPass) comp.addPass(this.extraPass);
    comp.addPass(new OutputPass());
    // SMAA stands in for MSAA where MSAA was skipped; disable completely on low quality and mobile (unless ultra)
    const allowSMAA = !this.touch && !isMobileOrTouch() && this.settings?.quality !== 'low';
    if (q.msaa && !this.samples && (allowSMAA || this.settings?.quality === 'ultra')) comp.addPass(new SMAAPass());
    r.shadowMap.enabled = this.settings.shadows !== false;
    this._w = w; this._h = h;
    this.grade.uniforms.uAspect.value = w / h;
  }

  // Install (or replace) the screen-FX post pass; kept across quality/setting rebuilds.
  setExtraPass(pass) {
    this.extraPass = pass;
    if (this.scene) this._buildComposer();
  }

  applySettings(settings) {
    const prevQ = this.q;
    this.settings = settings;
    this.touch = shouldUseTouch(settings);
    this.q = rendererQuality(settings, this.touch);
    const shadowChanged = this.renderer.shadowMap.enabled !== (settings.shadows !== false);
    if (prevQ !== this.q || shadowChanged) {
      if (prevQ !== this.q) this.dynScale = 1;
      this._buildComposer();
      this.scene?.traverse((o) => {
        if (o.material) {
          const m = Array.isArray(o.material) ? o.material : [o.material];
          const isLow = this.settings?.quality === 'low' || (this.touch && this.settings?.quality !== 'ultra');
          m.forEach((mm) => {
            if (mm.defines) {
              if (isLow) mm.defines.INK_QUALITY_LOW = 1;
              else delete mm.defines.INK_QUALITY_LOW;
            }
            mm.needsUpdate = true;
          });
        }
      });
    }
    if (this.bloom) this.bloom.enabled = !!(this.q.bloom && settings.bloom);
  }

  // Lowest dynamic scale: never below 0.75 of CSS-pixel density on desktop.
  // On mobile / touch devices, scale smoothly down to 0.55 when frame times exceed 20ms (<50 FPS).
  dynFloor() {
    if (this.touch) return 0.55;
    const base = Math.min(window.devicePixelRatio || 1, this.q.pixelRatio);
    return Math.max(0.5, Math.min(0.75, 0.75 / base));
  }

  // Dynamic resolution (never on ultra): scale the render density between dynFloor() and 1 of the quality preset.
  setDynamicScale(s) {
    s = Math.max(this.dynFloor(), Math.min(1, s));
    if (Math.abs(s - this.dynScale) < 0.01) return;
    this.dynScale = s;
    const pr = Math.min(window.devicePixelRatio || 1, this.q.pixelRatio) * s;
    this.renderer.setPixelRatio(pr);
    if (this.composer) {
      this.composer.setPixelRatio(pr);
      this.composer.setSize(this._w, this._h);
    }
  }

  resize(force = false) {
    const el = this.container || (typeof document !== 'undefined' ? document.getElementById('app') : null) || (typeof document !== 'undefined' ? document.body : null);
    const cw = el?.clientWidth || 0, ch = el?.clientHeight || 0;
    const ww = typeof window !== 'undefined' ? (window.innerWidth || 1) : 1;
    const wh = typeof window !== 'undefined' ? (window.innerHeight || 1) : 1;
    const w = cw > 10 ? cw : ww;
    const h = ch > 10 ? ch : wh;

    if (!force && w === this._w && h === this._h) return;
    this._w = w; this._h = h;
    this.renderer.setSize(w, h, false);
    if (this.renderer.domElement) {
      this.renderer.domElement.style.width = '100%';
      this.renderer.domElement.style.height = '100%';
    }
    if (this.composer) this.composer.setSize(w, h);
    this.gtao?.setSize(w, h);
    if (this.grade) this.grade.uniforms.uAspect.value = w / h;
    if (this.camera) { this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  }

  _updateDirectFx() {
    if (!this._directFxEl && this.container) {
      const el = document.createElement('div');
      el.id = 'iw-direct-fx';
      el.style.position = 'absolute';
      el.style.left = '0';
      el.style.top = '0';
      el.style.width = '100%';
      el.style.height = '100%';
      el.style.pointerEvents = 'none';
      el.style.zIndex = '2';
      el.style.display = 'none';
      this.container.appendChild(el);
      this._directFxEl = el;
    }
    if (!this._directFxEl) return;
    const hurt = this.grade?.uniforms?.uHurt?.value || 0;
    const flash = this.grade?.uniforms?.uFlash?.value || 0;
    if (hurt <= 0.01 && flash <= 0.01) {
      if (this._directFxEl.style.display !== 'none') this._directFxEl.style.display = 'none';
      return;
    }
    this._directFxEl.style.display = 'block';
    if (flash > 0.01 && hurt > 0.01) {
      const fAlpha = Math.min(0.8, flash);
      const hAlpha = Math.min(0.7, hurt * 0.6);
      this._directFxEl.style.background = `radial-gradient(circle, rgba(255,255,255,${fAlpha}) 40%, rgba(255,30,50,${hAlpha}) 100%)`;
    } else if (flash > 0.01) {
      const fAlpha = Math.min(0.85, flash);
      this._directFxEl.style.background = `rgba(255, 255, 255, ${fAlpha})`;
    } else {
      const hAlpha = Math.min(0.75, hurt * 0.65);
      this._directFxEl.style.background = `radial-gradient(circle, transparent 45%, rgba(255, 30, 50, ${hAlpha}) 100%)`;
    }
  }

  render() {
    this.resize();
    // colour grade recommended by the environment theme (day / dusk)
    const gr = G.env && G.env.grade;
    if (gr && gr !== this._gradeSrc && this.grade) {
      this._gradeSrc = gr;
      const u = this.grade.uniforms;
      for (const k of ['uSat', 'uVib', 'uContrast', 'uLift', 'uVignette', 'uExposure']) u[k].value = gr[k] ?? GradeShader.uniforms[k].value;
      u.uShadowTint.value.set(...(gr.uShadowTint || [0.975, 0.99, 1.035]));
      u.uHighTint.value.set(...(gr.uHighTint || [1.025, 1.0, 0.972]));
      // bloom per theme: [strength, radius, threshold] — dusk lets lamps / lit windows bloom, daylight keeps it to the sun
      const bl = gr.bloom || BLOOM;
      this.bloom.strength = bl[0]; this.bloom.radius = bl[1]; this.bloom.threshold = bl[2];
    }
    // Direct render fast-path: render straight to the default framebuffer when on low / mobile without extra post passes
    const isMobile = this.touch || isMobileOrTouch();
    if (this.useDirectRender && (!this.extraPass || !this.extraPass.enabled || isMobile || this.q?.directRender)) {
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.scene, this.camera);
      this._updateDirectFx();
      return;
    }
    if (this._directFxEl) this._directFxEl.style.display = 'none';
    this.composer.render();
  }
}
