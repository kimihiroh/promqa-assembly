// Handles API calls between React and Flask.

const API_BASE = "http://localhost:5050";  // hard-coded

export async function fetchData(annotatorId, idx) {
    const response = await fetch(
        `${API_BASE}/get_data?annotatorId=${annotatorId}&idx=${idx}`
    );
    const data = await response.json();
    return data;
}

export async function sendData(data) {
    const response = await fetch(
        `${API_BASE}/send_data`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        }
    );
    return response.json();
}
