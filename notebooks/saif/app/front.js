fetch("http://localhost:8000/predict/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ feature_1: "value1", feature_2: 10, feature_3: "GO" })
})
.then(response => response.json())
.then(data => console.log("Predictions:", data))
.catch(error => console.error("Error:", error));
