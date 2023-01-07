#! /usr/bin/env python3
#
# Copyright (C) 2020  Fx Bricks
# This file is part of the legocad python module.
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# LDraw Model conversion script

import os
import pymesh
import numpy as np
from numpy.linalg import norm
from scipy import spatial
from ldrawpy import *
from cqkit import *


scriptdir = os.path.dirname(os.path.realpath(__file__))

srcdir = os.path.normpath(scriptdir + os.path.sep + "../cad")
outdir = os.path.normpath(scriptdir + os.path.sep + "../ldraw")
subdir = os.path.normpath(scriptdir + os.path.sep + "../ldraw/s")

MIN_RES = 0.1
CIRCLE_RES = 24
CURVE_RES = 8

files = [
    "R40",
    "R56",
    "R64P",
    "R72",
    "R88",
    "R104",
    "R120",
    "R136",
    "R152",
    "S1.6",
    "S3.2",
    "S4",
    "S8",
    "S16",
    "S32",
]


def log_mesh(mesh, msg=None, edges=None):
    s = msg if msg is not None else ""
    es = "edges=%-5d " % (len(edges)) if edges is not None else ""
    print(
        "Mesh:  triangles=%-5d vertices=%-5d %s%s"
        % (len(mesh.vertices), len(mesh.faces), es, s)
    )


def fix_mesh(mesh):
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    log_mesh(mesh, "Remove duplicates")
    mesh, __ = pymesh.collapse_short_edges(mesh, MIN_RES, preserve_feature=True)
    log_mesh(mesh, "Collapse short edges")
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 170, 100)
    log_mesh(mesh, "Remote obtuse faces")
    mesh, __ = pymesh.collapse_short_edges(mesh, MIN_RES, preserve_feature=True)
    log_mesh(mesh, "Collapse short edges")
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    log_mesh(mesh, "Remove degenerate faces")
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    log_mesh(mesh, "Remove self intersections")
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    log_mesh(mesh, "New hull, remove duplicates")
    mesh, __ = pymesh.remove_isolated_vertices(mesh)
    log_mesh(mesh, "Remove isolated vertices")
    return mesh


def ldr_header(fn, prefix=""):
    head, tail = os.path.split(fn)
    h = LDRHeader()
    h.author = "L-Gauge.org"
    item = tail.replace(".dat", "").replace("RCTrack", "")
    if "R" in item:
        r = item.replace("R", "")
        h.title = prefix + " %s Curve track %s stud radius" % (item, r)
    else:
        sl = item.replace("S", "")
        h.title = prefix + " %s Straight track %s stud length" % (item, sl)
    h.file = tail
    h.name = tail
    s = []
    s.append(str(h))
    s.append("0 !LDRAW_ORG Unofficial_Part\n")
    s.append("0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt\n")
    s.append("0 BFC CERTIFY CCW\n")
    return "".join(s)


for f in files:
    fnstl = os.path.normpath(srcdir + os.sep + "RC%s.stl" % (f))
    fnstep = os.path.normpath(srcdir + os.sep + "RC%s.step" % (f))
    fnout = os.path.normpath(outdir + os.sep + "RCTrack%s.dat" % (f))

    print("Importing %s..." % (f))
    mesh = pymesh.load_mesh(fnstl)
    log_mesh(mesh, "Imported mesh")
    mesh = fix_mesh(mesh)
    mv = []
    for v in mesh.vertices:
        mv.append(Vector(tuple(v)))
    obj = import_step_file(fnstep)
    edges = obj.edges().vals()
    log_mesh(mesh, msg="Imported STEP", edges=edges)

    print("Discretizing edges...")
    edges = discretize_all_edges(
        edges, curve_res=CURVE_RES, circle_res=CIRCLE_RES, as_pts=True
    )
    log_mesh(mesh, edges=edges)

    vertices = np.array(mesh.vertices)
    epts = []
    for e in edges:
        e0 = list(e[0])
        e1 = list(e[1])
        p0 = tuple(vertices[spatial.distance.cdist([e0], vertices).argmin()])
        p1 = tuple(vertices[spatial.distance.cdist([e1], vertices).argmin()])
        match0 = abs(Vector(p0) - Vector(tuple(e0))) < 0.2
        match1 = abs(Vector(p1) - Vector(tuple(e1))) < 0.2
        if not (match0 and match1):
            if not match0:
                p0 = e0
            if not match1:
                p1 = e1
        if not abs(Vector(tuple(p0)) - Vector(tuple(p1))) < 0.025:
            epts.append((p0, p1))

    log_mesh(mesh, edges=epts)
    ldr_obj = mesh_to_ldr(mesh.faces, mv, LDR_DEF_COLOUR, epts, LDR_OPT_COLOUR)
    print(len(ldr_obj))
    hs = ldr_header(fnout, prefix="RCTrack")
    f = open(fnout, "w")
    f.write(hs)
    f.write(ldr_obj)
    f.write("0 NOFILE\n")
    f.close()
