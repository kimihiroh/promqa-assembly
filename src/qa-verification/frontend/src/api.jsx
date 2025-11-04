// Handles API calls between React and Flask.

const API_BASE = "http://localhost:5350";  // hard-coded

export async function fetchData(filename, idx) {
    const response = await fetch(
        `${API_BASE}/get_data?filename=${filename}&idx=${idx}`
    );
    const data = await response.json();
    return data;
}

export async function sendData(data) {
    console.log(data);
    const response = await fetch(
        `${API_BASE}/send_data`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        }
    );
    return response.json();
}
