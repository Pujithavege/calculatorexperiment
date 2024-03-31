DOM:
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Date-based Document Styling</title>
<style>
    body {
        font-family: Arial, sans-serif;
    }
</style>
</head>
<body>
<form id="dateForm">
    <label for="dateInput">Enter a Date:</label>
    <input type="date" id="dateInput" name="dateInput">
    <button type="button" onclick="changeDocumentStyles()">Apply Styles</button>
</form>

<script>
function changeDocumentStyles() {
    var dateInput = document.getElementById('dateInput').value;
    var currentDate = new Date();
    var inputDate = new Date(dateInput);
    
    // Check if the input date is today
    if (inputDate.toDateString() === currentDate.toDateString()) {
        document.body.style.color = 'green';
        document.body.style.fontSize = '20px';
    } else {
        document.body.style.color = 'black'; // Default color
        document.body.style.fontSize = '16px'; // Default font size
    }
}
</script>
</body>
</html>

