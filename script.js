document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = new FormData(this);

    try {
        let response = await fetch("http://127.0.0.1:5000/predict", {  // Update with your backend URL
            method: "POST",
            body: formData
        });

        let data = await response.json();

        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerText =
                "Patient: " + data.patient_name + " (Age: " + data.patient_age + ") - Prediction: " + data.prediction;
        }
    } catch (error) {
        document.getElementById("result").innerText = "Error connecting to server.";
    }
});
