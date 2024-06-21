from Catalogue import Catalogue

def main():
    # Initialize the Catalogue instance
    newCatalogue = Catalogue()

    # Define the parameters and directory path
    parameters = '/cluster/scratch/nmunro/parameters.yaml'
    path = '/cluster/scratch/nmunro/tpc5File/LBQ-20220331-I-BESND_286.tpc5'

    # Load data
    newCatalogue.loadData(path=path, isDataFile=True, num_batches=10)

    # Apply quakephase
    output = newCatalogue.applyQuakephase(parameters=parameters, maxWorkers=32, parallelProcessing=True)

    # Save results
    newCatalogue.saveData(quakePhaseOutput=output, fileName='tpc5Output')

if __name__ == "__main__":
    main()
