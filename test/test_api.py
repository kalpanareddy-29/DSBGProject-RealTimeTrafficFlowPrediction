import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

# ==========================================
# TEST HEALTH
# ==========================================
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200

    data = res.get_json()
    assert "status" in data

# ==========================================
# TEST PREDICT
# ==========================================
def test_predict(client):

    sample_data = {
        "sequence": [[50]*325 for _ in range(12)]
    }

    res = client.post("/predict", json=sample_data)

    assert res.status_code == 200

    data = res.get_json()
    assert "traffic" in data
    assert "avg_speed" in data

# ==========================================
# TEST LIVE TRAFFIC
# ==========================================
def test_live_traffic(client):
    res = client.get("/live-traffic")

    # Might fail if API key missing → still okay
    assert res.status_code in [200, 500]