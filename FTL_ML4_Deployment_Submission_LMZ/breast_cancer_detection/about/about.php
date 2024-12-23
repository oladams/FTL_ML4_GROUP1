<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Breast Cancer Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            color: #333;
        }

        header {
            background-color: #5d0477;
            color: white;
            padding: 1rem;
            text-align: center;
        }

        header h1 {
            margin: 0;
        }

        section {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .section-title {
            text-align: center;
            margin-bottom: 2rem;
            color: #5d0477;
        }

        .steps {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .step {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            flex: 1;
            min-width: 280px;
            max-width: 360px;
            text-align: center;
        }

        .step img {
            width: 60px;
            height: 60px;
            margin-bottom: 1rem;
        }

        .step h3 {
            color: #5d0477;
            margin-bottom: 0.5rem;
        }

        .call-to-action {
            text-align: center;
            margin-top: 2rem;
        }

        .call-to-action a {
            text-decoration: none;
            background-color: #5d0477;
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        .call-to-action a:hover {
            background-color: #430556;
        }

        @media (max-width: 768px) {
            .steps {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>About Breast Cancer Prediction</h1>
    </header>

    <section>
        <h2 class="section-title">How Our Prediction System Works</h2>
        <div class="steps">
            <div class="step">
                <img src="https://via.placeholder.com/60" alt="Data Collection">
                <h3>Step 1: Data Collection</h3>
                <p>We securely gather RNA data from patients, ensuring reliability and confidentiality.</p>
            </div>

            <div class="step">
                <img src="https://via.placeholder.com/60" alt="Analysis">
                <h3>Step 2: Machine Learning Analysis</h3>
                <p>Our advanced models process the data to detect potential cancer risks with high precision.</p>
            </div>

            <div class="step">
                <img src="https://via.placeholder.com/60" alt="Reporting">
                <h3>Step 3: Reporting and Insights</h3>
                <p>We provide actionable insights through detailed reports to aid decision-making.</p>
            </div>
        </div>

        <div class="call-to-action">
            <a href="#">Start a Prediction</a>
        </div>
    </section>
</body>
</html>
