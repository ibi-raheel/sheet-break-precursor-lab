import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { C } from '../lib/colors'
import {
  minutes, values, onsets, onsetMin, isFirst, masks, hazard, hazardAlarm, hazMax, yLo, yHi,
  TOTAL_MIN, N, clock, rowAt, r, f3, ruleAlarm, RULE, edges,
} from '../lib/data'
import { useWidth, useReducedMotion, useInView } from '../lib/hooks'
import { Chevron } from './Icons'

const H = 254
const L = { top: 10, trace: 132, hazTop: 150, hazH: 42, alarmY: 200, ruleY: 211, alarmH: 6, axis: 236 }
const BOTTOM = L.ruleY + L.alarmH

function setupCanvas(cv: HTMLCanvasElement, w: number, h: number) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr)
  cv.style.width = w + 'px'; cv.style.height = h + 'px'
  const ctx = cv.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return ctx
}

const yOf = (v: number) => {
  const c = Math.max(yLo, Math.min(yHi, v))
  return L.top + (1 - (c - yLo) / (yHi - yLo)) * (L.trace - L.top)
}

// Start on the first episode with three or more breaks.
const firstCluster = (() => {
  for (let k = 0; k + 2 < onsetMin.length; k++) if (onsetMin[k + 2] - onsetMin[k] < 240) return k
  return 0
})()

export default function TrendStrip() {
  const [wrap, w] = useWidth<HTMLDivElement>(900)
  const rm = useReducedMotion()
  const [inRef, seen] = useInView<HTMLDivElement>('-10% 0px')
  const win = w < 640 ? 360 : 720
  const [center, setCenter] = useState(onsetMin[firstCluster] + 160)
  const [showPen, setShowPen] = useState(true)
  const [showAlarm, setShowAlarm] = useState(true)
  const [showRule, setShowRule] = useState(true)
  const [hover, setHover] = useState<number | null>(null)
  const det = useRef<HTMLCanvasElement>(null)
  const ov = useRef<HTMLCanvasElement>(null)
  const anim = useRef<number>(0)
  const clampC = useCallback((c: number) => Math.max(win / 2, Math.min(TOTAL_MIN - win / 2, c)), [win])
  const c = clampC(center)
  const t0 = c - win / 2, t1 = c + win / 2
  const pxPerMin = w / win
  const X = (m: number) => (m - t0) * pxPerMin

  // Nearest break to the window's focus point (70% across).
  const focusK = useMemo(() => {
    const f = t0 + win * 0.7
    let best = 0, bd = Infinity
    onsetMin.forEach((m, k) => { const d = Math.abs(m - f); if (d < bd) { bd = d; best = k } })
    return best
  }, [t0, win])

  const goTo = useCallback((target: number) => {
    cancelAnimationFrame(anim.current)
    const from = c, to = clampC(target)
    if (rm || Math.abs(to - from) > win * 6) { setCenter(to); return }
    const t = performance.now(), dur = 340
    const step = (now: number) => {
      const p = Math.min(1, (now - t) / dur)
      const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2
      setCenter(from + (to - from) * e)
      if (p < 1) anim.current = requestAnimationFrame(step)
    }
    anim.current = requestAnimationFrame(step)
  }, [c, clampC, rm, win])

  const jump = (dir: 1 | -1) => {
    const focus = t0 + win * 0.7
    let k = dir === 1 ? onsetMin.findIndex((m) => m > focus + 1) : -1
    if (dir === -1) for (let j = onsetMin.length - 1; j >= 0; j--) if (onsetMin[j] < focus - 1) { k = j; break }
    if (k < 0) return
    goTo(onsetMin[k] + win * 0.2)
  }

  // ---- detail canvas
  useEffect(() => {
    const cv = det.current; if (!cv) return
    const ctx = setupCanvas(cv, w, H)
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, H)
    // chart-paper grid
    const minorStep = pxPerMin * 10 >= 7 ? 10 : 30
    for (let m = Math.ceil(t0 / minorStep) * minorStep; m <= t1; m += minorStep) {
      const x = Math.round(X(m)) + 0.5
      const major = m % 60 === 0
      ctx.strokeStyle = major ? C.gridMajor : C.gridMinor
      ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(x, L.top); ctx.lineTo(x, BOTTOM); ctx.stroke()
    }
    for (let i = 0; i <= 5; i++) {
      const y = Math.round(L.top + (i * (L.trace - L.top)) / 5) + 0.5
      ctx.strokeStyle = i === 0 || i === 5 ? C.gridMajor : C.gridMinor
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }
    for (let i = 0; i <= 2; i++) {
      const y = Math.round(L.hazTop + (i * L.hazH) / 2) + 0.5
      ctx.strokeStyle = i === 1 ? C.gridMinor : C.gridMajor
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }
    const i0 = Math.max(0, rowAt(t0) - 1), i1 = Math.min(N - 1, rowAt(t1) + 1)
    // restart masks
    ctx.fillStyle = C.mask
    masks.forEach(([s, e]) => {
      if (minutes[Math.min(e, N - 1)] < t0 || minutes[s] > t1) return
      const xa = X(minutes[s]), xb = X(minutes[Math.min(e, N - 1)] + 2)
      ctx.fillRect(xa, L.top, xb - xa, BOTTOM - L.top)
    })
    // timestamp gaps (line down, inferred)
    for (let i = Math.max(1, i0); i <= i1; i++) {
      if (minutes[i] - minutes[i - 1] > 2) {
        const xa = X(minutes[i - 1] + 2), xb = X(minutes[i])
        ctx.fillStyle = C.down; ctx.fillRect(xa, L.top, xb - xa, L.trace - L.top)
        ctx.save(); ctx.beginPath(); ctx.rect(xa, L.top, xb - xa, L.trace - L.top); ctx.clip()
        ctx.strokeStyle = 'rgba(200,54,29,0.22)'; ctx.lineWidth = 1
        for (let x = xa - 140; x < xb; x += 6) { ctx.beginPath(); ctx.moveTo(x, L.trace); ctx.lineTo(x + 122, L.top); ctx.stroke() }
        ctx.restore()
      }
    }
    // trace (break the pen at gaps)
    ctx.strokeStyle = C.trace; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'
    ctx.beginPath()
    for (let i = i0; i <= i1; i++) {
      const x = X(minutes[i]), y = yOf(values[i])
      if (i === i0 || minutes[i] - minutes[i - 1] > 2) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.stroke()
    // hazard pen
    if (showPen) {
      const hy = (p: number) => L.hazTop + L.hazH - (p / hazMax) * L.hazH
      ctx.beginPath()
      for (let i = i0; i <= i1; i++) { const x = X(minutes[i]); i === i0 ? ctx.moveTo(x, hy(hazard[i])) : ctx.lineTo(x, hy(hazard[i])) }
      ctx.lineTo(X(minutes[i1]), L.hazTop + L.hazH); ctx.lineTo(X(minutes[i0]), L.hazTop + L.hazH); ctx.closePath()
      ctx.fillStyle = C.hazardFill; ctx.fill()
      ctx.beginPath()
      for (let i = i0; i <= i1; i++) { const x = X(minutes[i]); i === i0 ? ctx.moveTo(x, hy(hazard[i])) : ctx.lineTo(x, hy(hazard[i])) }
      ctx.strokeStyle = C.hazard; ctx.lineWidth = 1.5; ctx.stroke()
    }
    const lane = (a: Uint8Array, y: number, col: string) => {
      ctx.fillStyle = col
      let s = -1
      for (let i = i0; i <= i1 + 1; i++) {
        const on = i <= i1 && a[i]
        if (on && s < 0) s = i
        if (!on && s >= 0) { const xa = X(minutes[s]), xb = X(minutes[i - 1] + 2); ctx.fillRect(xa, y, Math.max(1.5, xb - xa), L.alarmH); s = -1 }
      }
    }
    // empty lane rails
    ctx.strokeStyle = C.gridMajor; ctx.lineWidth = 1
    ;[L.alarmY, L.ruleY].forEach((y) => { ctx.beginPath(); ctx.moveTo(0, y + L.alarmH / 2 + 0.5); ctx.lineTo(w, y + L.alarmH / 2 + 0.5); ctx.stroke() })
    if (showAlarm) lane(hazardAlarm, L.alarmY, C.hazard)
    if (showRule) lane(ruleAlarm, L.ruleY, C.rule)
    // breaks
    onsets.forEach((row, k) => {
      const m = minutes[row]; if (m < t0 - 5 || m > t1 + 5) return
      const x = Math.round(X(m)) + 0.5
      ctx.strokeStyle = C.brk; ctx.lineWidth = isFirst[k] ? 1.75 : 1.25
      ctx.setLineDash(isFirst[k] ? [] : [4, 3])
      ctx.beginPath(); ctx.moveTo(x, L.top + 9); ctx.lineTo(x, BOTTOM); ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = C.brk
      ctx.beginPath(); ctx.moveTo(x - 4.5, L.top); ctx.lineTo(x + 4.5, L.top); ctx.lineTo(x, L.top + 8); ctx.closePath(); ctx.fill()
    })
    // axis
    ctx.fillStyle = C.ink2; ctx.font = '500 11px "JetBrains Mono", ui-monospace, monospace'; ctx.textBaseline = 'top'
    const lab = win <= 360 ? 60 : 120
    for (let m = Math.ceil(t0 / lab) * lab; m <= t1; m += lab) {
      const x = X(m); const txt = m % 1440 === 0 ? clock(m).slice(0, -6) : clock(m, false)
      const tw = ctx.measureText(txt).width
      if (x - tw / 2 < 2 || x + tw / 2 > w - 2) continue
      ctx.fillText(txt, x - tw / 2, L.axis - 12)
    }
    // hover crosshair
    if (hover !== null) {
      const x = Math.round(X(minutes[hover])) + 0.5
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.globalAlpha = 0.55
      ctx.beginPath(); ctx.moveTo(x, L.top); ctx.lineTo(x, BOTTOM); ctx.stroke()
      ctx.globalAlpha = 1
      ctx.fillStyle = C.trace; ctx.strokeStyle = C.paper; ctx.lineWidth = 2
      ctx.beginPath(); ctx.arc(x, yOf(values[hover]), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke()
    }
  }, [w, t0, t1, win, pxPerMin, showPen, showAlarm, showRule, hover])

  // ---- overview canvas (whole 28 days)
  const OVH = 40
  useEffect(() => {
    const cv = ov.current; if (!cv) return
    const ctx = setupCanvas(cv, w, OVH)
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, w, OVH)
    const per = TOTAL_MIN / w
    const mn = new Float32Array(w).fill(Infinity), mx = new Float32Array(w).fill(-Infinity)
    for (let i = 0; i < N; i++) {
      const px = Math.min(w - 1, Math.floor(minutes[i] / per))
      const v = Math.max(yLo, Math.min(yHi, values[i]))
      if (v < mn[px]) mn[px] = v; if (v > mx[px]) mx[px] = v
    }
    const oy = (v: number) => 4 + (1 - (v - yLo) / (yHi - yLo)) * (OVH - 8)
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.globalAlpha = 0.7
    ctx.beginPath()
    for (let x = 0; x < w; x++) if (mn[x] !== Infinity) { ctx.moveTo(x + 0.5, oy(mx[x])); ctx.lineTo(x + 0.5, oy(mn[x]) + 0.5) }
    ctx.stroke(); ctx.globalAlpha = 1
    ctx.strokeStyle = C.brk; ctx.lineWidth = 1
    onsetMin.forEach((m) => { const x = Math.round(m / per) + 0.5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 7); ctx.stroke() })
    for (let d = 1; d < 28; d++) {
      const x = Math.round((d * 1440) / per) + 0.5
      ctx.strokeStyle = C.gridMajor; ctx.beginPath(); ctx.moveTo(x, OVH - 6); ctx.lineTo(x, OVH); ctx.stroke()
    }
    const xa = t0 / per, xb = t1 / per
    ctx.fillStyle = 'rgba(27,33,41,0.08)'; ctx.fillRect(xa, 0, xb - xa, OVH)
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(xa + 0.75, 0.75, xb - xa - 1.5, OVH - 1.5)
  }, [w, t0, t1])

  // ---- pointer: drag to pan the detail, drag/click on overview to move the window
  const drag = useRef<{ x: number; c: number; id: number } | null>(null)
  const onDetDown = (e: React.PointerEvent) => {
    if (drag.current) return
    cancelAnimationFrame(anim.current)
    drag.current = { x: e.clientX, c, id: e.pointerId }
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }
  const onDetMove = (e: React.PointerEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const mx = e.clientX - rect.left
    if (drag.current && drag.current.id === e.pointerId) {
      setCenter(clampC(drag.current.c - (e.clientX - drag.current.x) / pxPerMin))
    }
    if (e.pointerType === 'mouse') setHover(rowAt(t0 + mx / pxPerMin))
  }
  const onUp = (e: React.PointerEvent) => { if (drag.current?.id === e.pointerId) drag.current = null }
  const ovDrag = useRef(false)
  const ovSet = (e: React.PointerEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    setCenter(clampC(((e.clientX - rect.left) / rect.width) * TOTAL_MIN))
  }

  const hv = hover !== null && minutes[hover] >= t0 && minutes[hover] <= t1 ? hover : null
  const hvX = hv !== null ? X(minutes[hv]) : 0
  const k = focusK
  const epN = useMemo(() => { let n = 0; for (let j = 0; j <= k; j++) if (isFirst[j]) n++; return n }, [k])

  return (
    <div className="strip" ref={inRef}>
      <div className="strip-head">
        <div className="strip-read mono" aria-live="polite">
          <span className="brk-dot" aria-hidden="true" />
          Break {k + 1} of {r.meta.breaks} · {clock(onsetMin[k])} · {isFirst[k] ? `first of episode ${epN}` : `follow-on, ${onsetMin[k] - onsetMin[k - 1]} min after the last`}
        </div>
        <div className="strip-nav">
          <button className="btn" onClick={() => jump(-1)} aria-label="Previous break"><Chevron dir="left" /> Previous break</button>
          <button className="btn" onClick={() => jump(1)} aria-label="Next break">Next break <Chevron dir="right" /></button>
        </div>
      </div>
      <div ref={wrap} className={`strip-paper ${seen ? 'drawn' : ''}`}>
        <canvas
          ref={det}
          className="strip-detail"
          role="img"
          aria-label={`Chart recorder trace of sensor ${r.timeline.signal} from ${clock(t0)} to ${clock(t1)}, breaks marked in red`}
          onPointerDown={onDetDown}
          onPointerMove={onDetMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={() => setHover(null)}
        />
        <div className="lane-tag mono" style={{ top: L.top + 2 }}>CH1 {r.timeline.signal}</div>
        <div className={`lane-tag mono ${showPen ? '' : 'off'}`} style={{ top: L.hazTop + 2, color: showPen ? C.hazard : undefined }}>CH2 hazard, next 20 min{showPen ? '' : ': off'}</div>
        <div className={`lane-tag lane-sm mono ${showAlarm ? '' : 'off'}`} style={{ top: L.alarmY - 3, color: showAlarm ? C.hazard : undefined }}>ALM hazard</div>
        <div className={`lane-tag lane-sm mono ${showRule ? '' : 'off'}`} style={{ top: L.ruleY - 2, color: showRule ? C.rule : undefined }}>ALM {r.timeline.signal}</div>
        {hv !== null && (
          <div className="tip mono" style={{ left: Math.min(Math.max(hvX + 12, 0), w - 190), top: 18 }}>
            <div>{clock(minutes[hv])}</div>
            <div>{r.timeline.signal} = {values[hv].toFixed(1)}</div>
            {showPen && <div style={{ color: C.hazard }}>hazard {f3(hazard[hv])}</div>}
          </div>
        )}
      </div>
      <canvas
        ref={ov}
        className="strip-overview"
        role="slider"
        tabIndex={0}
        aria-label="Position in the 28-day record"
        aria-valuemin={0}
        aria-valuemax={Math.round(TOTAL_MIN)}
        aria-valuenow={Math.round(c)}
        aria-valuetext={clock(c)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') { e.preventDefault(); setCenter(clampC(c + win / 2)) }
          if (e.key === 'ArrowLeft') { e.preventDefault(); setCenter(clampC(c - win / 2)) }
        }}
        onPointerDown={(e) => { ovDrag.current = true; (e.target as Element).setPointerCapture(e.pointerId); ovSet(e) }}
        onPointerMove={(e) => { if (ovDrag.current) ovSet(e) }}
        onPointerUp={() => { ovDrag.current = false }}
      />
      <div className="strip-foot">
        <div className="channels" role="group" aria-label="Recorder channels">
          <label className="chan"><input type="checkbox" checked={showPen} onChange={(e) => setShowPen(e.target.checked)} /><span className="sw" style={{ ['--c' as string]: C.hazard }} />Hazard pen</label>
          <label className="chan"><input type="checkbox" checked={showAlarm} onChange={(e) => setShowAlarm(e.target.checked)} /><span className="sw bar" style={{ ['--c' as string]: C.hazard }} />Hazard alarm</label>
          <label className="chan"><input type="checkbox" checked={showRule} onChange={(e) => setShowRule(e.target.checked)} /><span className="sw bar" style={{ ['--c' as string]: C.rule }} />Sensor rule alarm</label>
        </div>
        <ul className="legend mono" aria-label="Legend">
          <li><i className="lg lg-brk" />first break</li>
          <li><i className="lg lg-brk dash" />follow-on</li>
          <li><i className="lg lg-down" />line down (gap)</li>
          <li><i className="lg lg-mask" />restart mask</li>
        </ul>
      </div>
      <AlarmLog t0={t0} t1={t1} haz={showAlarm} rule={showRule} />
    </div>
  )
}

function AlarmLog({ t0, t1, haz, rule }: { t0: number; t1: number; haz: boolean; rule: boolean }) {
  const i0 = rowAt(t0), i1 = Math.min(N - 1, rowAt(t1))
  type Line = { m: number; tag: string; txt: string; cls: string }
  const lines: Line[] = []
  onsets.forEach((row, k) => {
    const m = minutes[row]; if (m < t0 || m > t1) return
    lines.push({ m, tag: 'BRK', txt: isFirst[k] ? 'sheet break, first of episode' : `follow-on, +${m - onsetMin[k - 1]} min`, cls: 'l-brk' })
  })
  if (haz) edges(hazardAlarm, i0, i1).forEach((i) => lines.push({ m: minutes[i], tag: 'HAZ', txt: `hazard high, p ${f3(hazard[i])}`, cls: 'l-haz' }))
  if (rule) edges(ruleAlarm, i0, i1).forEach((i) => lines.push({ m: minutes[i], tag: 'RUL', txt: `${RULE.signal} below ${RULE.threshold.toFixed(1)} (${values[i].toFixed(1)})`, cls: 'l-rul' }))
  lines.sort((a, b) => a.m - b.m)
  return (
    <div className="alarmlog mono" aria-label="Alarm log for the visible window">
      <div className="al-head"><span>ALARM LOG</span><span>{clock(t0)} to {clock(t1, false)}</span></div>
      <ol>
        {lines.length === 0 && <li className="l-none">no events in this window</li>}
        {lines.map((l, j) => (
          <li key={j} className={l.cls}><time>{clock(l.m, false)}</time><b>{l.tag}</b><span>{l.txt}</span></li>
        ))}
      </ol>
    </div>
  )
}
