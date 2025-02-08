function predict() {
    let data = {
        AGE: parseInt(document.getElementById("age").value),
        TOTAL_VOTES: parseInt(document.getElementById("total_votes").value),
        GENERAL_VOTES: parseInt(document.getElementById("general_votes").value),
        POSTAL_VOTES: parseInt(document.getElementById("postal_votes").value),
        TOTAL_ELECTORS: parseInt(document.getElementById("total_electors").value),
        CRIMINAL_CASES: parseInt(document.getElementById("criminal_cases").value),
        ASSETS: parseFloat(document.getElementById("assets").value),
        LIABILITIES: parseFloat(document.getElementById("liabilities").value),
        EDUCATION: parseInt(document.getElementById("education").value),
        CATEGORY: parseInt(document.getElementById("category").value),
        GENDER: parseInt(document.getElementById("gender").value),
        PARTY: parseInt(document.getElementById("party").value)
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById("result").innerText = `Prediction: ${result.Prediction}`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error in prediction!";
    });
}
