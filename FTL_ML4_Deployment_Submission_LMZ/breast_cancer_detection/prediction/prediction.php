<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Prediction</title>
    <link rel="stylesheet" href="prediction.css">
</head>
<body>
<header>
    <div class="logo">Breast Cancer</div>
    <nav>
        <ul class="menu">
            <li><a href="../main.php">Home</a></li>
            <li><a href="">Prediction</a></li>
            <li><a href="../about/about.php">About</a></li>
            <li><a href="../contact/contact.php">Contact us</a></li>
        </ul>
    </nav>
</header>
    <section id="prediction">
        <div class="container">
            <h2>Breast Cancer Prediction</h2>
            <p>Enter the details below to predict the risk of breast cancer based on provided data.</p>

            <form id="prediction-form">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" placeholder="Enter your age" required>
                </div>

                <div class="form-group">
                    <label for="rna">RNA Data:</label>
                    <input type="file" id="rna" name="rna" accept=".csv, .txt" required>
                </div>

                <div class="form-group">
                    <label for="history">Medical History (optional):</label>
                    <textarea id="history" name="history" placeholder="Enter any relevant medical history"></textarea>
                </div>

                <div class="cta">
                    <button type="submit" class="btn">Submit</button>
                </div>
            </form>

            <div id="prediction-result" class="result">
                <h3>Prediction Result</h3>
                <p id="result-text">Your prediction result will appear here.</p>
            </div>

            <div id="prediction-explanation">
                <h4>How It Works:</h4>
                <p>Our algorithm analyzes RNA data and age to calculate the risk of breast cancer. Based on the data provided, the model generates a prediction result that helps in early detection.</p>
            </div>
        </div>
    </section>
    <footer>
        <p>Contact us for more information or support.</p>
        <a href="#">Privacy Policy</a> | <a href="#">Contact</a>
    </footer>
</body>
</html>
