using System;

namespace NS_ANN
{

    /// <summary>
    /// La seguente classe è corretta, ma non ottimizzata.
    /// </summary>

    class RPropPlus
    {
        const double c1 = 0.4, c2 = 1.2, m = 0.8;
        double Error;
        double[] desired_output;
        double[][] dterr;
        double[][][] delta;

        //Calcola la variazione dei pesi sinaptici da applicare a ciascun collegamento sinaptico.
        //Per ogni neurone interno sommo gli errori delle SUE SINAPSI e basta.
        private double Delta(FFANN ffann, int j, int k, int q)
        {
            double _delta = 0;
            if (j == ffann.NumLayers - 2)
                _delta = ffann.layer[j+1].act.Derr1(desired_output[q], ffann.layer[j+1].perceptron[q].getAction());
            else
            {
                if(dterr[j + 1][q] != 0)
                    _delta = dterr[j + 1][q] * ffann.layer[j+1].act.Dcompute(ffann.layer[j + 1].perceptron[q].getAction());
                else
                    _delta = (dterr[j + 1][q] + 1) * ffann.layer[j + 1].act.Dcompute(ffann.layer[j + 1].perceptron[q].getAction());

            }
            dterr[j][k] += _delta * ffann.layer[j].perceptron[k].getSynapsys(q);
            if(delta[j][k][q] * _delta  <= 0) //se il segno della derivata rimane uguale per due correzioni di fila, aumentiamo la velocità nel verso opposto.
                return (c1 * _delta * ffann.layer[j].perceptron[k].getAction());
            else
                return (c2 * _delta * ffann.layer[j].perceptron[k].getAction() + m * delta[j][k][q] /*Momentum*/);
        }

        //Costruttore, si occupa dell'allenamento vero e proprio.
        public RPropPlus(FFANN ffann, double[][] stdDataset)
        {
            int epochs = 0;
            desired_output = new double[ffann.NumPercept[ffann.NumLayers - 1]];
            
            //inizializzazione di delta[][][].
            delta = new double[ffann.NumLayers - 1][][]; //Matrice delta che conterrà le correzione dei pesi da applicare alla fine della propagazione.
            for (int i = 0; i < ffann.NumLayers - 1; i++)
                delta[i] = new double[ffann.NumPercept[i]][];
            for (int j = 0; j < delta.Length; j++)
                for (int i = 0; i < delta[j].Length; i++) //Calcolo il numero totale di sinapsi nella rete.
                    delta[j][i] = new double[ffann.NumPercept[j + 1]];
            //Fine inizializzazione delta[][][].

            //Inizializzo ed azzero la matrice dterr.
            dterr = new double[ffann.NumLayers - 1][];
            for (int i = 0; i < ffann.NumLayers - 1; i++)
            {
                dterr[i] = new double[ffann.NumPercept[i]];
                for (int z = 0; z < ffann.NumPercept[i]; z++)
                    dterr[i][z] = 0;
            }
            //Fine azzeramento.

            do //Scorro le epoche in cui la rete si allenerà.
            {
                Error = 0;
                for (int i = 0; i < stdDataset.Length; i++) //Scorro tutti i samples nel dataset.
                {
                    

                    ffann.Predict(stdDataset[i]); //Calcolo le uscite della rete dato il sample i.

                    //Copio i valori desiderati in un vettore comodo e calcolo l'Errore Quadratico Medio.
                    for (int j = 0; j < ffann.NumPercept[ffann.NumLayers - 1]; j++)
                    {
                        desired_output[j] = stdDataset[i][ffann.NumPercept[0] + j];
                        Error += ffann.layer[ffann.NumLayers - 1].act.Err(desired_output[j], ffann.layer[ffann.NumLayers - 1].perceptron[j].getAction());
                    }
                    //End

                    //Calcolo i delta dei pesi sinaptici.
                    for (int j = ffann.NumLayers - 2; j >= 0; j--) //Scorro la rete dal penultimo layer al primo, per avere accesso diretto ai pesi sinaptici da correggere.
                        for (int k = 0; k < ffann.NumPercept[j]; k++) // Scorro i percettroni dello strato j.
                            for (int q = 0; q < ffann.NumPercept[j + 1]; q++) // Scorro le sinapsi.
                                delta[j][k][q] = Delta(ffann, j, k, q);

                    //Setto i nuovi pesi applicando i delta dei pesi sinaptici.
                    for (int j = 0; j < ffann.NumLayers - 1; j++)
                        for (int k = 0; k < ffann.NumPercept[j]; k++)
                            for (int q = 0; q < ffann.NumPercept[j + 1]; q++)
                                ffann.layer[j].perceptron[k].setSynapsys(ffann.layer[j].perceptron[k].getSynapsys(q) - delta[j][k][q], q);

                    //è necessario azzerare dterr per ogni sample analizzato.
                    for (int j = 0; j < dterr.Length; j++)
                        for (int z = 0; z < dterr[j].Length; z++)
                            dterr[j][z] = 0;
                    //Fine azzeramento.
                }

                //Stampa l'errore quadratico medio se if è true.
                epochs++;
                if (epochs % 100 == 0)
                    Console.WriteLine("\nErrore Quadratico Medio: " + Error + "\n");
                
                if (Error <= 11)
                    Console.Write("\nEpoche: " + epochs + "\n" + "\n Errore Quadratico Medio: " + Error + "\n");
            } while ((epochs != 400) && (Error > 11));
        }
    }


    class Genetic
    {
        double Fitness;

        public Genetic(FFANN ffann, double[][] stdDataset)
        {
            Random Rand = new Random();
            Fitness = Rand.NextDouble();
        }


        private void Cross(FFANN ffann)
        {

        }

        private void Mutate(FFANN ffann)
        {

        }

        private void Reproduce()
        {

        }

    }

    
    class Particle_swarm_optimization
    {
        double Cohesion, Separation;

        public Particle_swarm_optimization(FFANN ffann, double[][] stdDataset)
        {
            Random Rand = new Random();
            Cohesion = Rand.NextDouble();
            Separation = Rand.NextDouble();
        }
    }

    
    class Rica
    {
        public Rica()
        {

        }
    }


    //Per ogni grande categoria di problemi nel Machine Learning usa 
    //l'output stabile nella versione con exclude
    //per allenare una rete neurale profonda.
    class Equilizer
    {
        public Equilizer(FFANN ffann)
        {

        }






        public void Shape(FFANN ffann)
        {

        }






        private void Kill(Layer layer, int index) //Delete layer.perceptron[index].
        {

        }

        private void Cut(Perceptron neuron, int index) //Delete neuron.Synapses[index]
        {

        }

        private void Gen(Layer layer, bool[] parameters, int index) //Create layer.perceptron[index] with 'parameters' specified.
        {

        }

        private void Bind(Perceptron neuron1, Perceptron neuron2) //Create neuron1 - neuron2 synapsys. 
        {

        }

        private void Exclude(Layer layer, int index)//Exclude layer.perceptron[index] from computation.
        {

        }
    }
}

