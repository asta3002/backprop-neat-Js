// Import the necessary module
var DataSet = require('./dataset.js');
var fs = require('fs');

// Generate each dataset
var circleData = DataSet.generateRandomData(0);
var xorData = DataSet.generateRandomData(1);
var gaussiansData = DataSet.generateRandomData(2);
var spiralData = DataSet.generateRandomData(3);

// Combine all datasets into a single object
var allData = {
    circle: circleData,
    xor: xorData,
    gaussians: gaussiansData,
    spiral: spiralData
};

// Convert the data to a JSON string
var jsonString = JSON.stringify(allData, null, 2);

// Save the JSON string to a file named 'datasets.json'
fs.writeFile('datasets.json', jsonString, (err) => {
    if (err) {
        console.error('Error writing file:', err);
    } else {
        console.log('Successfully saved all datasets to datasets.json');
    }
});
