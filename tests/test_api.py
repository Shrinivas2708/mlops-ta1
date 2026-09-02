"""End-to-end API test: a good image must PASS, a defective one must be DEFECT."""
import cv2
from fastapi.testclient import TestClient

import scripts.generate_sample_data as gen
from app.main import app

client = TestClient(app)


def encode(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True


def test_good_part_passes():
    with TestClient(app) as c:
        img = gen.augment(gen.base_part())
        r = c.post("/inspect", files={"file": ("g.png", encode(img), "image/png")})
        assert r.status_code == 200
        assert r.json()["verdict"] == "PASS"


def test_defect_part_flagged():
    with TestClient(app) as c:
        img = gen.augment(gen.add_defect(gen.base_part()))
        r = c.post("/inspect", files={"file": ("d.png", encode(img), "image/png")})
        assert r.status_code == 200
        assert r.json()["verdict"] == "DEFECT"
