"""Cut each animal photo into rigged parts for SVG puppets.

Coordinates are fractions of the cut-out image (x/W, y/H). Every part gets its own
transparent PNG; the body has the lifted regions removed and the joint area inpainted
so a turning head reveals fur, not a hole. Output: assets/animals/<animal>-<part>.webp
plus a rig.json describing boxes, pivots, parent/child order and eyelids.
"""
import json, os
import numpy as np, cv2
from PIL import Image

# Usage: put background-removed cut-outs at <SRC>/cut-<animal>.png (e.g. via `rembg` with the isnet-general-use model),
# then run `python3 rig.py <SRC>`. Requires numpy, opencv-python-headless, pillow.
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + '/source'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'animals')
os.makedirs(OUT, exist_ok=True)
H = 380            # working/export height in px (all rig coords are in this space)

RIGS = {
 'deer': {
   'parts': [
     {'id':'head','parent':None,'pivot':(.62,.50),'poly':[(.33,0),(1,0),(1,.36),(.80,.36),(.75,.44),(.70,.50),(.56,.50),(.51,.42),(.46,.34),(.35,.31),(.33,.2)]},
     {'id':'earL','parent':'head','pivot':(.55,.285),'poly':[(.39,.205),(.52,.215),(.58,.27),(.56,.315),(.47,.315),(.40,.27)]},
     {'id':'earR','parent':'head','pivot':(.73,.285),'poly':[(.715,.27),(.78,.225),(.88,.205),(.87,.26),(.80,.315),(.72,.315)]},
   ],
   'eyes':[(.585,.325,.022),(.728,.325,.022)], 'inpaint':40},
 'fox': {
   'parts': [
     {'id':'tail','parent':None,'pivot':(.84,.47),'poly':[(.77,.40),(.92,.40),(1,.58),(1,1),(.79,1),(.77,.74)]},
     {'id':'legFar','parent':None,'pivot':(.37,.56),'poly':[(.31,.53),(.44,.53),(.44,.90),(.33,.90)]},
     {'id':'legNear','parent':None,'pivot':(.25,.56),'poly':[(.18,.53),(.31,.52),(.31,.89),(.21,.89)]},
     {'id':'head','parent':None,'pivot':(.27,.36),'poly':[(0,0),(.33,0),(.35,.16),(.33,.36),(.24,.44),(.06,.44),(0,.37)]},
     {'id':'earL','parent':'head','pivot':(.08,.15),'poly':[(0,0),(.115,0),(.135,.12),(.07,.175),(0,.13)]},
     {'id':'earR','parent':'head','pivot':(.24,.13),'poly':[(.185,.05),(.25,0),(.31,0),(.30,.12),(.21,.145)]},
   ],
   'eyes':[(.142,.232,.018),(.236,.226,.016)], 'inpaint':36},
 'bear': {
   'parts': [
     {'id':'foot','parent':None,'pivot':(.885,.88),'poly':[(.86,.74),(1,.74),(1,.99),(.86,.99)]},
     {'id':'head','parent':None,'pivot':(.27,.42),'poly':[(.10,0),(.41,0),(.47,.16),(.46,.36),(.33,.44),(.12,.42),(.07,.2)]},
     {'id':'earL','parent':'head','pivot':(.17,.095),'poly':[(.12,.02),(.22,.02),(.23,.11),(.13,.12)]},
     {'id':'earR','parent':'head','pivot':(.345,.06),'poly':[(.30,0),(.39,0),(.39,.08),(.31,.08)]},
   ],
   'eyes':[(.335,.19,.012),(.408,.183,.01)], 'inpaint':40},
 'squirrel': {
   'parts': [
     {'id':'tail','parent':None,'pivot':(.64,.78),'poly':[(.56,.02),(1,.02),(1,.32),(.88,.96),(.66,.96),(.57,.56)]},
     {'id':'head','parent':None,'pivot':(.24,.47),'poly':[(0,.15),(.27,.17),(.31,.30),(.27,.50),(.17,.57),(.03,.56),(0,.40)]},
     {'id':'ear','parent':'head','pivot':(.19,.28),'poly':[(.14,.17),(.25,.17),(.27,.28),(.16,.30)]},
     {'id':'paws','parent':None,'pivot':(.30,.66),'poly':[(0,.56),(.20,.57),(.38,.66),(.30,.76),(.14,.79),(0,.79)]},
   ],
   'eyes':[(.105,.425,.022)], 'inpaint':30},
 'rabbit': {
   'parts': [
     {'id':'head','parent':None,'pivot':(.50,.44),'poly':[(.12,.07),(.72,.09),(.82,.30),(.68,.45),(.30,.45),(.12,.36)]},
     {'id':'earB','parent':'head','pivot':(.74,.22),'poly':[(.65,0),(.96,0),(.92,.18),(.71,.25)]},
     {'id':'earF','parent':'head','pivot':(.56,.21),'poly':[(.47,0),(.69,0),(.63,.20),(.50,.23)]},
   ],
   'eyes':[(.488,.243,.026)], 'inpaint':36},
 'owl': {
   'parts': [
     {'id':'wing','parent':None,'pivot':(.50,.24),'poly':[(.44,.16),(.76,.34),(.96,.60),(.94,.96),(.70,.93),(.48,.55)]},
     {'id':'head','parent':None,'pivot':(.25,.32),'poly':[(0,0),(.50,0),(.50,.29),(.32,.35),(.05,.34),(0,.22)]},
   ],
   'eyes':[(.118,.105,.05),(.31,.103,.05)], 'inpaint':40, 'lidTone':'dark'},
 'raccoon': {
   'parts': [
     {'id':'paw','parent':None,'pivot':(.50,.83),'poly':[(.42,.67),(.63,.67),(.63,.85),(.42,.85)]},
     {'id':'head','parent':None,'pivot':(.50,.64),'poly':[(.03,0),(.97,0),(.97,.64),(.03,.64)]},
     {'id':'earL','parent':'head','pivot':(.19,.23),'poly':[(.07,.05),(.28,.05),(.27,.25),(.09,.26)]},
     {'id':'earR','parent':'head','pivot':(.64,.23),'poly':[(.54,.04),(.74,.04),(.74,.25),(.55,.25)]},
   ],
   'eyes':[(.285,.39,.022),(.495,.39,.022)], 'inpaint':30},
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
    im = Image.open(f'{SRC}/cut-{animal}.png').convert('RGBA')
    w = round(im.width*H/im.height)
    rgba = np.array(im.resize((w, H), Image.LANCZOS)).astype(np.float32)
    # same green despill + feet feather as before
    a = rgba[..., 3]
    edge = cv2.dilate((a < 250).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    lim = np.maximum(rgba[..., 0], rgba[..., 2])*1.02 + 4
    rgba[..., 1] = np.where(edge & (rgba[..., 1] > lim), lim, rgba[..., 1])
    if animal not in ('owl',):
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
    rig_out[animal] = {'w': w, 'h': H, 'body': bbox, 'parts': parts_out, 'lids': lids}

json.dump(rig_out, open(f'{OUT}/rig.json', 'w'), indent=1)
print(json.dumps({k: [p['id'] for p in v['parts']] for k, v in rig_out.items()}))
