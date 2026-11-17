
document.getElementById('assess-btn').addEventListener('click', async function(e) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = '';
    resultDiv.innerHTML = '<em>Assessing...</em>';

    const formData = new FormData();
    const image = document.getElementById('image').files[0];
    const criteria = document.getElementById('criteria').value;
    const mode = document.getElementById('mode').value;
    const max_marks = document.getElementById('max_marks').value;

    formData.append('image', image);
    formData.append('criteria', criteria);
    formData.append('mode', mode);
    formData.append('max_marks', max_marks);

    try {
        const response = await fetch('http://localhost:8000/assess', {
            method: 'POST',
            body: formData
        });
        let data;
        try {
            data = await response.json();
        } catch (jsonErr) {
            resultDiv.textContent = 'Error: Could not parse server response.';
            return;
        }
        if (data.marks !== undefined && data.feedback !== undefined) {
            resultDiv.innerHTML = `<b>Marks:</b> ${data.marks}<br><b>Feedback:</b><br>${data.feedback}`;
        } else if (data.result) {
            resultDiv.textContent = data.result;
        } else {
            resultDiv.textContent = 'Unexpected response from server.';
        }
    } catch (err) {
        resultDiv.textContent = 'Network or server error: ' + err;
    }
    return false;
});
