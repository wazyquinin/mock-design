"""Cut each animal photo into rigged parts for SVG puppets.

Coordinates are fractions of the cut-out image (x/W, y/H). Every part gets its own
transparent PNG; the body has the lifted regions removed and the joint area inpainted
so a turning head reveals fur, not a hole. Output: assets/animals/<animal>-<part>.webp
plus a rig.json describing boxes, pivots, parent/child order and eyelids.
"""
import json, os
import numpy as np, cv2
from PIL import Image

# Usage: put background-removed cut-outs in <SRC> (sitting/owl poses) and <SRC>/walk (side-on walking poses),
# e.g. via `rembg` with the isnet-general-use model, then run `python3 rig.py <SRC>`.
# Requires numpy, opencv-python-headless, pillow.
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + '/source'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'animals')
os.makedirs(OUT, exist_ok=True)
H = 380            # working/export height in px (all rig coords are in this space)

WALK = SRC + '/walk'
RIGS = {
 'deer': {'src': WALK+'/cut-deer.png',
   'parts': [
     {'id':'tail','parent':None,'pivot':(.06,.27),'poly':[(0,.22),(.09,.22),(.10,.36),(0,.38)]},
     {'id':'hind','parent':None,'pivot':(.10,.56),'poly':[(0,.52),(.19,.52),(.17,.99),(0,.99)]},
     {'id':'frontFar','parent':None,'pivot':(.54,.57),'poly':[(.48,.54),(.60,.54),(.60,.99),(.49,.99)]},
     {'id':'frontNear','parent':None,'pivot':(.665,.56),'poly':[(.61,.52),(.73,.50),(.87,.97),(.84,1),(.73,1),(.62,.72)]},
     {'id':'head','parent':None,'pivot':(.74,.42),'poly':[(.68,0),(1,0),(1,.30),(.88,.38),(.80,.48),(.70,.48),(.66,.30)]},
     {'id':'earL','parent':'head','pivot':(.79,.12),'poly':[(.69,0),(.80,0),(.82,.13),(.74,.15),(.69,.08)]},
     {'id':'earR','parent':'head','pivot':(.905,.12),'poly':[(.89,.02),(1,0),(1,.08),(.95,.14),(.89,.14)]},
   ],
   'eyes':[(.83,.168,.012),(.885,.158,.01)], 'inpaint':40, 'gain':2.2, 'tint':(1.32, 1.0, .72)},
 'fox': {'src': WALK+'/cut-fox.png',
   'parts': [
     {'id':'tail','parent':None,'pivot':(.71,.24),'poly':[(.67,.10),(.80,.12),(1,.40),(1,.76),(.82,.66),(.68,.36)]},
     {'id':'hindFar','parent':None,'pivot':(.585,.55),'poly':[(.52,.50),(.65,.50),(.63,.96),(.53,.96)]},
     {'id':'frontFar','parent':None,'pivot':(.285,.57),'poly':[(.12,.62),(.27,.52),(.34,.56),(.31,.66),(.21,.99),(.11,.99)]},
     {'id':'hindNear','parent':None,'pivot':(.705,.47),'poly':[(.64,.42),(.78,.44),(.81,1),(.73,1),(.68,.68)]},
     {'id':'frontNear','parent':None,'pivot':(.37,.53),'poly':[(.32,.48),(.43,.48),(.45,1),(.36,1),(.32,.74)]},
     {'id':'head','parent':None,'pivot':(.20,.36),'poly':[(0,.04),(.23,.03),(.26,.30),(.21,.48),(.12,.68),(0,.66)]},
     {'id':'earL','parent':'head','pivot':(.10,.19),'poly':[(.05,.04),(.135,.04),(.14,.21),(.06,.22)]},
     {'id':'earR','parent':'head','pivot':(.19,.14),'poly':[(.155,.03),(.215,.03),(.225,.16),(.165,.17)]},
   ],
   'eyes':[], 'inpaint':34},
 'bear': {'src': WALK+'/cut-bear.png',
   'parts': [
     {'id':'hindFar','parent':None,'pivot':(.07,.58),'poly':[(0,.54),(.13,.54),(.11,.93),(0,.93)]},
     {'id':'frontFar','parent':None,'pivot':(.42,.55),'poly':[(.33,.50),(.51,.50),(.46,.99),(.33,.99)]},
     {'id':'hindNear','parent':None,'pivot':(.18,.53),'poly':[(.09,.48),(.29,.48),(.28,.99),(.09,.99)]},
     {'id':'frontNear','parent':None,'pivot':(.66,.51),'poly':[(.59,.46),(.74,.46),(.89,.99),(.72,1),(.61,.72)]},
     {'id':'head','parent':None,'pivot':(.69,.36),'poly':[(.66,0),(.84,0),(1,.18),(1,.42),(.84,.47),(.68,.47),(.62,.16)]},
     {'id':'ear','parent':'head','pivot':(.78,.11),'poly':[(.73,0),(.83,0),(.83,.13),(.74,.13)]},
   ],
   'eyes':[(.865,.188,.009)], 'inpaint':40},
 'raccoon': {'src': WALK+'/cut-raccoon.png',
   'parts': [
     {'id':'tail','parent':None,'pivot':(.66,.18),'poly':[(.62,.10),(.80,.13),(1,.62),(.97,.76),(.80,.46),(.62,.30)]},
     {'id':'hindFar','parent':None,'pivot':(.635,.57),'poly':[(.58,.52),(.71,.52),(.71,.93),(.59,.93)]},
     {'id':'frontFar','parent':None,'pivot':(.33,.66),'poly':[(.04,.78),(.20,.67),(.33,.62),(.37,.70),(.21,.83),(.05,.87)]},
     {'id':'hindNear','parent':None,'pivot':(.50,.57),'poly':[(.43,.52),(.57,.52),(.57,.99),(.41,.99)]},
     {'id':'frontNear','parent':None,'pivot':(.38,.64),'poly':[(.24,.77),(.35,.60),(.43,.63),(.37,.81),(.25,.85)]},
     {'id':'head','parent':None,'pivot':(.18,.50),'poly':[(0,.34),(.13,.30),(.21,.38),(.23,.62),(.11,.76),(0,.74)]},
     {'id':'ear','parent':'head','pivot':(.10,.52),'poly':[(.06,.34),(.125,.34),(.13,.54),(.06,.54)]},
   ],
   'eyes':[(.018,.60,.007)], 'inpaint':30},
 'rabbit': {'src': WALK+'/cut-rabbit.png',
   'parts': [
     {'id':'hindFoot','parent':None,'pivot':(.74,.88),'poly':[(.42,.86),(.82,.84),(.84,1),(.42,1)]},
     {'id':'frontPaws','parent':None,'pivot':(.30,.80),'poly':[(.18,.78),(.42,.78),(.42,1),(.16,1)]},
     {'id':'head','parent':None,'pivot':(.38,.52),'poly':[(0,.26),(.42,.22),(.47,.45),(.42,.62),(.15,.64),(0,.58)]},
     {'id':'ears','parent':'head','pivot':(.36,.31),'poly':[(.24,0),(.53,0),(.46,.32),(.30,.35)]},
   ],
   'eyes':[(.212,.42,.02)], 'inpaint':36},
 'squirrel': {'src': SRC+'/cut-squirrel.png',
   'parts': [
     {'id':'tail','parent':None,'pivot':(.64,.78),'poly':[(.56,.02),(1,.02),(1,.32),(.88,.96),(.66,.96),(.57,.56)]},
     {'id':'head','parent':None,'pivot':(.24,.47),'poly':[(0,.15),(.27,.17),(.31,.30),(.27,.50),(.17,.57),(.03,.56),(0,.40)]},
     {'id':'ear','parent':'head','pivot':(.19,.28),'poly':[(.14,.17),(.25,.17),(.27,.28),(.16,.30)]},
     {'id':'paws','parent':None,'pivot':(.30,.66),'poly':[(0,.56),(.20,.57),(.38,.66),(.30,.76),(.14,.79),(0,.79)]},
   ],
   'eyes':[(.105,.425,.022)], 'inpaint':30, 'extra':{'leap': WALK+'/cut-squirrel.png'}},
 'owl': {'src': SRC+'/cut-owl.png',
   'parts': [
     {'id':'wing','parent':None,'pivot':(.50,.24),'poly':[(.44,.16),(.76,.34),(.96,.60),(.94,.96),(.70,.93),(.48,.55)]},
     {'id':'head','parent':None,'pivot':(.25,.32),'poly':[(0,0),(.50,0),(.50,.29),(.32,.35),(.05,.34),(0,.22)]},
   ],
   'eyes':[(.118,.105,.05),(.31,.103,.05)], 'inpaint':40, 'lidTone':'dark', 'noFeather':True},
}

def poly_mask(poly, w, h, feather=1.5):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([(x*w, y*h) for x, y in poly], np.int32)], 255)
    return cv2.GaussianBlur(m.astype(np.float32)/255., (0, 0), feather)

def save_part(name, rgba, w, h):
    a = rgba[..., 3]
    ys, xs = np.where(a > 6)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    crop = Image.fromarray(rgba[y0:y1, x0:x1].astype(np.uint8), 'RGBA')
    crop.save(f'{OUT}/{name}.webp', quality=86, method=6)
    return [int(x0), int(y0), int(x1-x0), int(y1-y0)]

rig_out = {}
for animal, spec in RIGS.items():
    im = Image.open(spec['src']).convert('RGBA')
    w = round(im.width*H/im.height)
    rgba = np.array(im.resize((w, H), Image.LANCZOS)).astype(np.float32)
    if spec.get('gain'):
        lin = (rgba[..., :3]/255.)**2.2*spec['gain']*np.array(spec.get('tint', (1, 1, 1))); rgba[..., :3] = np.clip(lin**(1/2.2)*255, 0, 255)
    # same green despill + feet feather as before
    a = rgba[..., 3]
    edge = cv2.dilate((a < 250).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    lim = np.maximum(rgba[..., 0], rgba[..., 2])*1.02 + 4
    rgba[..., 1] = np.where(edge & (rgba[..., 1] > lim), lim, rgba[..., 1])
    if not spec.get('noFeather'):
        ramp = np.clip((H-1-np.arange(H))/(H*0.06), 0, 1)[:, None]; rgba[..., 3] *= ramp
    alpha = rgba[..., 3]/255.
    sil = alpha > .05

    # parts: top-level lifted regions are removed from the body; children are removed from their parent
    lifted = np.zeros((H, w), np.float32)
    masks = {p['id']: poly_mask(p['poly'], w, H) for p in spec['parts']}
    child_union = {}
    for p in spec['parts']:
        if p['parent']: child_union[p['parent']] = np.maximum(child_union.get(p['parent'], 0), masks[p['id']])
    parts_out = []
    for p in spec['parts']:
        m = masks[p['id']].copy()
        if p['id'] in child_union: m = np.clip(m - child_union[p['id']], 0, 1)   # children are drawn on top; their base overlaps the parent
        part = rgba.copy(); part[..., 3] = rgba[..., 3]*m
        box = save_part(f'{animal}-{p["id"]}', part, w, H)
        if not p['parent']: lifted = np.maximum(lifted, masks[p['id']])
        parts_out.append({'id': p['id'], 'parent': p['parent'], 'pivot': [p['pivot'][0]*w, p['pivot'][1]*H], 'box': box})

    # body = everything not lifted; inpaint the joint band so moving parts reveal fur
    keep = np.clip(1 - lifted, 0, 1)
    body = rgba.copy(); body[..., 3] = rgba[..., 3]*keep
    remain = (body[..., 3] > 128).astype(np.uint8)
    dist = cv2.distanceTransform(1 - remain, cv2.DIST_L2, 5)
    band = (lifted > .02) & sil & (dist <= spec['inpaint'])
    rgb8 = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    holes = (((lifted > .02) & sil) | ~sil).astype(np.uint8)*255
    filled = cv2.inpaint(cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), holes, 7, cv2.INPAINT_TELEA)
    filled = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB).astype(np.float32)
    fall = np.clip(1 - dist/spec['inpaint'], 0, 1)**.7
    body[..., :3] = np.where(band[..., None], filled, body[..., :3])
    body[..., 3] = np.where(band, np.maximum(body[..., 3], 255*fall*(alpha > .3)), body[..., 3])
    bbox = save_part(f'{animal}-body', body, w, H)

    # eyelid colours: the fur just above each eye
    lids = []
    for ex, ey, r in spec['eyes']:
        cx, cy, rr = int(ex*w), int(ey*H), max(3, int(r*w))
        yy, xx = np.mgrid[0:H, 0:w]; dd = np.hypot(xx-cx, yy-cy)
        ann = (dd > rr*1.7) & (dd < rr*2.6) & (alpha > .6)
        col = np.median(rgba[ann][:, :3], axis=0) if ann.any() else np.array([90, 70, 50])
        col = col*(.7 if spec.get('lidTone') == 'dark' else .86)
        lids.append({'x': ex*w, 'y': ey*H, 'r': r*w, 'color': '#%02x%02x%02x' % tuple(int(c) for c in np.clip(col, 0, 255))})
    extra = {}
    for name, path in spec.get('extra', {}).items():
        e = Image.open(path).convert('RGBA'); ew = round(e.width*H*.62/e.height)     # leap pose ~ same body size
        e = e.resize((ew, round(H*.62)), Image.LANCZOS)
        ea = np.array(e).astype(np.float32); m1 = ea[..., 3] > 200; m0 = rgba[..., 3] > 200
        src = cv2.cvtColor(ea[..., :3].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref = cv2.cvtColor(np.clip(rgba[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            ms, ss = src[..., c][m1].mean(), src[..., c][m1].std()+1e-3; mr, sr = ref[..., c][m0].mean(), ref[..., c][m0].std()
            src[..., c] = (src[..., c]-ms)*(sr/ss)*.85 + mr
        ea[..., :3] = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        Image.fromarray(ea.astype(np.uint8), 'RGBA').save(f'{OUT}/{animal}-{name}.webp', quality=86, method=6)
        extra[name] = [ew, round(H*.62)]
    rig_out[animal] = {'w': w, 'h': H, 'body': bbox, 'parts': parts_out, 'lids': lids, 'extra': extra}

json.dump(rig_out, open(f'{OUT}/rig.json', 'w'), indent=1)
print(json.dumps({k: [p['id'] for p in v['parts']] for k, v in rig_out.items()}))
