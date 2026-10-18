// INKWAVE — Device Performance Classifier & Adaptive Graphics Governor.
// Automatically benchmarks and scores each individual device's GPU and system capacity (0.0 to 1.0)
// and dynamically scales resolution and quality so every player achieves a smooth 60 FPS experience.

let _cachedProfile = null;

function detectMobile() {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return false;
  if (window._dynamicTouchDetected) return true;
  const ua = navigator.userAgent || navigator.vendor || window.opera || '';
  const isIPad = (navigator.platform === 'MacIntel' || /iPad/i.test(ua)) && (navigator.maxTouchPoints || 0) > 1;
  const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|Tablet/i.test(ua)
    || Boolean(navigator?.userAgentData?.mobile)
    || isIPad;
  if (isMobileUA) return true;
  const isCoarse = typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
  const hasTouchPoints = (navigator.maxTouchPoints || 0) > 0;
  const isMobileViewport = typeof window.innerWidth === 'number' && (
    Math.min(window.innerWidth, window.innerHeight) <= 768 || window.innerWidth <= 768
  );
  return Boolean(isCoarse && hasTouchPoints && isMobileViewport);
}

/**
 * Inspect WebGL hardware strings, CPU concurrency, RAM, and display density to determine
 * a device performance profile and baseline factor (0.45 to 1.0).
 */
export function getDeviceProfile() {
  if (_cachedProfile) return _cachedProfile;

  let gpuRenderer = '';
  let gpuVendor = '';
  const isMobile = detectMobile();

  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) {
          gpuRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '';
          gpuVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || '';
        }
        if (!gpuRenderer) {
          gpuRenderer = gl.getParameter(gl.RENDERER) || '';
        }
      }
    } catch {
      /* ignore context creation issues */
    }
  }

  const gpu = (gpuRenderer + ' ' + gpuVendor).toLowerCase();
  const cores = typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4;
  const ram = typeof navigator !== 'undefined' ? (navigator.deviceMemory || 4) : 4;
  const dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;

  // GPU Tier Scoring
  // Tier 3 (Ultra / Enthusiast): High-end discrete GPUs
  const isUltraGpu = /rtx\s*(?:30[7-9]0|40[7-9]0|50[7-9]0)|rx\s*(?:6[89]00|7[89]00)|apple\s*m[2-4]\s*(?:pro|max|ultra)/i.test(gpu);

  // Tier 2 (High): Modern gaming desktop GPUs, Apple M1/M2/M3 base, high-end mobile
  const isHighGpu = isUltraGpu || /rtx|gtx\s*(?:1660|1070|1080|980)|rx\s*(?:580|590|5[567]00|6600|7600)|apple\s*m[1-4]|adreno\s*(?:7[3-9]\d|8\d\d)|mali-g7(?:8|9|1\d)/i.test(gpu);

  // Tier 1 (Medium): Mid-range laptops, Apple A-series mobile, Iris Xe, Vega, mainstream phones
  const isMedGpu = isHighGpu || /gtx|radeon|iris\s*xe|vega|apple\s*a1[4-8]|adreno\s*(?:6[6-9]\d|7[1-2]\d)|mali-g(?:6\d|7[1-7])/i.test(gpu);

  // Tier 0 (Low): Integrated Intel HD/UHD, low-power Mali/Adreno, software renderers
  const isLowGpu = /intel\s*(?:hd|uhd)\s*graphics|mali-g(?:3|5)\d|mali-4\d\d|adreno\s*(?:5\d\d|6[0-5]\d)|powervr|swiftshader|llvmpipe|software/i.test(gpu) || !isMedGpu;

  let tier = 'medium';
  let baselineFactor = 0.75;

  if (isMobile) {
    if (isHighGpu && ram >= 6 && cores >= 6) {
      tier = 'medium';
      baselineFactor = 0.85;
    } else if (isMedGpu && ram >= 4) {
      tier = 'low';
      baselineFactor = 0.70;
    } else {
      tier = 'low';
      baselineFactor = 0.55;
    }
  } else {
    // Desktop / Laptop
    // Laptops and battery-constrained systems have confined thermal dissipation;
    // Auto selects the balanced 'medium' preset (direct 1-pass render, native 1:1 pixels, 1024 shadows,
    // no GTAO raymarching or 4x MSAA thrashing) for locked 60 FPS and cool thermals.
    const hasBattery = typeof navigator !== 'undefined' && ('getBattery' in navigator || 'battery' in navigator);
    const isEnthusiastDesktop = isUltraGpu && ram >= 16 && cores >= 12 && !hasBattery;

    if (isEnthusiastDesktop) {
      tier = 'high';
      baselineFactor = 0.95;
    } else if (isHighGpu || (isMedGpu && !isLowGpu)) {
      tier = 'medium';
      baselineFactor = 0.85;
    } else {
      tier = 'low';
      baselineFactor = 0.65;
    }
  }

  // Clamping high-DPI screens on modest hardware to avoid fillrate bottlenecks
  if (dpr > 2.0 && baselineFactor < 0.9) {
    baselineFactor = Math.max(0.55, baselineFactor * 0.9);
  }

  _cachedProfile = {
    gpu: gpuRenderer || 'Default WebGL',
    tier,
    baselineFactor: +baselineFactor.toFixed(2),
    isMobile,
    cores,
    ram,
    dpr,
  };

  return _cachedProfile;
}

/**
 * Get recommended quality tier based on device benchmark.
 */
export function getAutoQualityTier() {
  return getDeviceProfile().tier;
}

/**
 * Performance Governor: Continuously tracks frame times and modulates dynamic resolution scaling
 * to maintain a steady 60 FPS target on any device.
 */
export class PerformanceGovernor {
  constructor(renderer) {
    this.R = renderer;
    const profile = getDeviceProfile();
    this.currentScale = profile.baselineFactor;
    this.minFloor = renderer?.dynFloor ? renderer.dynFloor() : (profile.isMobile ? 0.55 : 0.65);
    this.maxCeil = 1.0;
    this.acc = 0;
    this.n = 0;
    this.windowT = 0;
    this.headroomCount = 0;
    this.active = true;

    // Apply baseline factor to renderer on boot
    if (this.R?.setDynamicScale) {
      this.R.setDynamicScale(this.currentScale);
    }
  }

  /**
   * Called every frame in main game loop with frame delta (dt).
   * Evaluates rolling average frame time every 2.5 seconds.
   */
  update(dt, isPlaying, qualitySetting) {
    // Only automatically govern when in active gameplay and quality is 'auto' (or not explicitly locked)
    if (!this.active || !isPlaying || qualitySetting === 'ultra' || (qualitySetting === 'high' && !detectMobile())) {
      if (qualitySetting !== 'auto') return;
    }

    if (dt <= 0 || dt > 0.25) return;
    this.acc += dt;
    this.n++;
    this.windowT += dt;

    if (this.windowT < 2.5) return;

    const avgMs = (this.acc / this.n) * 1000;
    this.acc = 0;
    this.n = 0;
    this.windowT = 0;

    // Target: 60 FPS (16.6ms). Drop threshold: >19.5ms (<51 FPS).
    if (avgMs > 19.5 && this.currentScale > this.minFloor + 0.02) {
      this.currentScale = Math.max(this.minFloor, +(this.currentScale - 0.05).toFixed(2));
      this.headroomCount = 0;
      this.R?.setDynamicScale?.(this.currentScale);
    }
    // Headroom threshold: <14.0ms (>71 FPS headroom). Gradually step scale up if steady for 3 consecutive checks (7.5s).
    else if (avgMs < 14.0 && this.currentScale < this.maxCeil - 0.02) {
      this.headroomCount++;
      if (this.headroomCount >= 3) {
        this.currentScale = Math.min(this.maxCeil, +(this.currentScale + 0.05).toFixed(2));
        this.headroomCount = 0;
        this.R?.setDynamicScale?.(this.currentScale);
      }
    } else {
      this.headroomCount = 0;
    }
  }

  reset(factor = null) {
    if (factor != null) this.currentScale = factor;
    this.acc = 0;
    this.n = 0;
    this.windowT = 0;
    this.headroomCount = 0;
    this.R?.setDynamicScale?.(this.currentScale);
  }
}
