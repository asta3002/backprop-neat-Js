var R = require('./ml/recurrent.js');
var N = require('./ml/neat.js');
var DataSet = require('./dataset.js');
var fs = require('fs');

// Dataset configuration
const datasets = [
    { name: 'circle', id: 0, description: 'Circle classification dataset' },
    { name: 'xor', id: 1, description: 'XOR logic gate dataset' },
    { name: 'gaussian', id: 2, description: 'Gaussian clusters dataset' },
    { name: 'spiral', id: 3, description: 'Spiral classification dataset' }
];

// Function to convert R.Mat to regular array format
function matToArray(mat) {
    const result = [];
    for (let i = 0; i < mat.n; i++) {
        const row = [];
        for (let j = 0; j < mat.d; j++) {
            row.push(mat.get(i, j));
        }
        result.push(row);
    }
    return result;
}

// Function to convert label vector to array
function labelVecToArray(labelMat) {
    const result = [];
    for (let i = 0; i < labelMat.n; i++) {
        result.push(labelMat.w[i]);
    }
    return result;
}

// Function to generate and save dataset in JAX-compatible format
function generateAndSaveDataset(datasetConfig) {
    console.log(`Generating ${datasetConfig.name} dataset...`);
    
    // Generate the dataset
    DataSet.generateRandomData(datasetConfig.id);
    
    // Get training data
    const trainData = matToArray(DataSet.getTrainData());
    const trainLabels = labelVecToArray(DataSet.getTrainLabel());
    
    // Get test data
    const testData = matToArray(DataSet.getTestData());
    const testLabels = labelVecToArray(DataSet.getTestLabel());
    
    // Create dataset object in JAX-NEAT compatible format
    const dataset = {
        metadata: {
            name: datasetConfig.name,
            description: datasetConfig.description,
            input_dim: 2,
            output_dim: 1,
            num_classes: 2,
            train_size: DataSet.getTrainLength(),
            test_size: DataSet.getTestLength(),
            generated_at: new Date().toISOString()
        },
        train: {
            inputs: trainData,
            targets: trainLabels
        },
        test: {
            inputs: testData,
            targets: testLabels
        }
    };
    
    // Save as JSON file
    const filename = `${datasetConfig.name}_dataset.json`;
    fs.writeFileSync(filename, JSON.stringify(dataset, null, 2));
    console.log(`Saved ${filename}`);
    
    // Also save in NumPy-like format for easier JAX loading
    const npyFormat = {
        train_X: trainData,
        train_y: trainLabels,
        test_X: testData,
        test_y: testLabels
    };
    
    const npyFilename = `${datasetConfig.name}_dataset_npy.json`;
    fs.writeFileSync(npyFilename, JSON.stringify(npyFormat, null, 2));
    console.log(`Saved ${npyFilename} (NumPy-like format)`);
    
    return dataset;
}

// Function to generate sample batch for demonstration
function generateSampleBatch(datasetName, datasetId) {
    console.log(`\nGenerating sample batch for ${datasetName}...`);
    
    DataSet.generateRandomData(datasetId);
    DataSet.generateMiniBatch();
    
    const batchData = matToArray(DataSet.getBatchData());
    const batchLabels = labelVecToArray(DataSet.getBatchLabel());
    
    console.log(`Batch size: ${DataSet.getBatchLength()}`);
    console.log('Sample batch data (first 5 points):');
    for (let i = 0; i < Math.min(5, batchData.length); i++) {
        console.log(`  Input: [${batchData[i][0].toFixed(3)}, ${batchData[i][1].toFixed(3)}] -> Label: ${batchLabels[i]}`);
    }
}

// Function to create a combined dataset file
function createCombinedDataset(generatedDatasets) {
    const combined = {
        metadata: {
            description: 'Combined dataset containing all four classification problems',
            datasets: datasets.map(d => d.name),
            generated_at: new Date().toISOString()
        },
        datasets: {}
    };
    
    generatedDatasets.forEach((dataset, index) => {
        combined.datasets[datasets[index].name] = dataset;
    });
    
    fs.writeFileSync('combined_datasets.json', JSON.stringify(combined, null, 2));
    console.log('\nSaved combined_datasets.json');
}

// Main execution
function main() {
    console.log('=== JAX-NEAT Dataset Generator ===\n');
    
    const generatedDatasets = [];
    
    // Generate all datasets
    datasets.forEach(dataset => {
        const generated = generateAndSaveDataset(dataset);
        generatedDatasets.push(generated);
        
        // Show sample batch
        generateSampleBatch(dataset.name, dataset.id);
        console.log('---\n');
    });
    
    // Create combined dataset
    createCombinedDataset(generatedDatasets);
    
    // Summary
    console.log('=== Generation Complete ===');
    console.log('Generated files:');
    datasets.forEach(dataset => {
        console.log(`  - ${dataset.name}_dataset.json`);
        console.log(`  - ${dataset.name}_dataset_npy.json`);
    });
    console.log('  - combined_datasets.json');
    
    console.log('\nDataset statistics:');
    console.log(`  Training samples per dataset: ${DataSet.getTrainLength()}`);
    console.log(`  Test samples per dataset: ${DataSet.getTestLength()}`);
    console.log(`  Input dimensions: 2`);
    console.log(`  Output classes: 2 (binary classification)`);
}

// Export functions for potential reuse
module.exports = {
    generateAndSaveDataset,
    generateSampleBatch,
    createCombinedDataset,
    datasets
};

// Run if this is the main module
if (require.main === module) {
    main();
}