using System;


//Develope Occham Razor!!!


namespace NS_ANN
{
    
    public class FFANN
    {
        //Parametri della rete
        public short NumLayers = 3; //Numero di strati.
        public short[] NumPercept;  //Numero di Percettroni in ciascuno strato (in ordine da strato 0 a numLayers-1).
        const double Bias = 1;      //Costante euristica (ricavata in trial & error)
        public Layer[] layer;
        Random rand = new Random();

        //Costruttore, imposta i parametri essenziali della Rete Neurale.
        public FFANN(int numCols, int numOut)
        {
            NumPercept = new short[NumLayers];

            NumPercept[0] = (short)(numCols - 1);
            NumPercept[1] = 7;
            //NumPercept[2] = 5;
            //NumPercept[3] = 5;
            //NumPercept[4] = 5;
            //NumPercept[5] = 5;
            //NumPercept[6] = 5;
            //NumPercept[7] = 5;
            //NumPercept[8] = 5;
            //NumPercept[9] = 5;
            //NumPercept[10] = 5;
            NumPercept[NumLayers - 1] = (short)numOut;
            build();
        }

        //Inizializzo la rete neurale creando i vari strati e percettroni.
        private void build()
        {
            layer = new Layer[NumLayers];
            layer[0] = new Layer(0, this, rand, null);
            layer[1] = new Layer(1, this, rand, new LogisticSigmoid());
            //layer[2] = new Layer(2, this, rand, new HiperTan());
            //layer[3] = new Layer(3, this, rand, new HiperTan());
            //layer[4] = new Layer(4, this, rand, new HiperTan());
            //layer[5] = new Layer(5, this, rand, new HiperTan());
            //layer[6] = new Layer(6, this, rand, new HiperTan());
            //layer[7] = new Layer(7, this, rand, new HiperTan());
            //layer[8] = new Layer(8, this, rand, new HiperTan());
            //layer[9] = new Layer(9, this, rand, new HiperTan());
            //layer[10] = new Layer(10, this, rand, new HiperTan());
            layer[NumLayers - 1] = new Layer((short)(NumLayers -1), this, rand, new LogisticSigmoid());

        }
        //End Build

        //Dati i primi n-1 parametri, predicono l'ultimo.
        public void PredictShow(double[] data)
        {
            readData(data);
            for (int i = 0; i < NumLayers - 1; i++)
                propagate(i);
            layer[NumLayers - 1].showLayerAction();
            layer[NumLayers - 1].response();
        }

        public void Predict(double[] data)
        {
            readData(data);
            for (int i = 0; i < NumLayers - 1; i++)
                propagate(i);
        }
        //End Prediction.

        //inserisce n-1 parametri nel potenziale d'azione dei neuroni di input
        private void readData(double[] data)
        {
            for (int i = 0; i < NumPercept[0]; i++)
                layer[0].perceptron[i].setAction(data[i]);
        }

        //Dato lo strato di partenza, propaga il potenziale d'azione nello strato successivo.
        private void propagate(int i)
        {
            int j, k;
            double sum;                             //Accumulatore.
            for (j = 0; j < NumPercept[i + 1]; j++) //Scorro i percettroni dello strato successivo.
            {
                sum = 0; //Azzero l'accumulatore
                for (k = 0; k < NumPercept[i]; k++) //Scorro i percettroni dello strato di partenza.
                    sum += layer[i].perceptron[k].getAction() * layer[i].perceptron[k].getSynapsys(j); /* moltiplica potenziale d'azione 
                                                                                                        * per peso sinaptico della connessione 
                                                                                                        * fra k e j.*/
                sum += Bias;//Se non lo capisci non continuare a leggere e ristudiati la teoria prima.
                layer[i + 1].perceptron[j].setAction(layer[i + 1].act.Compute(sum)); /* Applica la funzione di costo e assegna il nuovo 
                                                                                 * potenziale d'azione.*/
            }
            /*per applicare il softmax, decommenta questo segmento.*/
            /*if(i == NumLayers - 2)
                act.Softmax(layer[i+1]);*/
        }
    }



    public class Layer
    {
        public short N; //numero identificativo layer
        public double b = 0.7;
        public Perceptron[] perceptron;
        public Activation act;
        public FFANN ffann;

        public Layer(short n, FFANN parent, Random rand, Activation value)
        {
            N = n;
            act = value;
            ffann = parent;
            perceptron = new Perceptron[ffann.NumPercept[N]];
            for (int i = 0; i < ffann.NumPercept[N]; i++)
                perceptron[i] = new Perceptron(i, this, rand);
        }

        public void showLayerAction()
        {
            Console.WriteLine("Layer " + N + ":\n");
            for (int i = 0; i < ffann.NumPercept[N]; i++)
                Console.WriteLine(perceptron[i].getAction() + "\n");
        }
        public void response()
        {
            int max = 0;
            for (int i = 1; i < perceptron.Length; i++)
                if (perceptron[i].getAction() > perceptron[max].getAction())
                    max = i;
            switch (max){
                case 0:
                    Console.WriteLine("Iris-Virginica");
                    break;
                case 1:
                    Console.WriteLine("Iris-Versicolor");
                    break;
                case 2:
                    Console.WriteLine("Iris-Setosa");
                    break;
                /*case 3:
                    Console.WriteLine("Ragazzo");
                    break;*/
            }
            
        }
        public void setLayerAction(Layer layer)
        {
            for (int i = 0; i < ffann.NumPercept[N]; i++)
                perceptron[i].setAction(layer.perceptron[i].getAction());
        }
    }



    sealed public class Perceptron
    {
        Layer _layer;
        private double[] synapsys; //Peso di ciascun collegamento neurone-neurone fra strati consecutivi
        private double action = 0; //Valore passato durante Train/Prediction
        public int NumSynapses;
        int nOrder; //Indica l'indice del percettrone nel vettore perceptron del corrispettivo layer.


        //Metodi protezione dati
        public double getSynapsys(int i)
        {
            if ((i < 0) || (i >= NumSynapses))
                return 1;
            return synapsys[i];
        }

        public void setSynapsys(double weight, int i)
        {
            if ((i < 0) || (i >= NumSynapses))
                return;
            synapsys[i] = weight;
        }

        public void setAction(double data)
        {
            action = data;
        }

        public double getAction()
        {
            return action;
        }

        public int getNOrder()
        {
            return nOrder;
        }
        //end protection methods

        //Costruttore di percettroni, inizializza i pesi sinaptici con valori casuali
        public Perceptron(int i, Layer parent, Random rand)
        {

            _layer = parent;

            if (_layer.N < _layer.ffann.NumLayers - 1)
            {
                NumSynapses = _layer.ffann.NumPercept[_layer.N + 1];
                synapsys = new double[NumSynapses];
            }
            else
                NumSynapses = 0;

            nOrder = i;
            for (int j = 0; j < NumSynapses; j++)
                synapsys[j] = rand.NextDouble() - 0.5;//Pesi sinaptici di Nguyen: [-0.5, 0.5].

            if((NumSynapses == 0) && (i == _layer.ffann.NumPercept[_layer.ffann.NumLayers - 1] - 1))
            {
                Layer[] Layers = _layer.ffann.layer;
                double sum = 0;
                _layer.b = _layer.b * Math.Pow(_layer.ffann.NumPercept[1], (1 / _layer.ffann.NumPercept[0]));

                for (int k = 0; k < Layers[0].perceptron.Length; k++)
                {
                    sum = 0;
                    foreach (double j in Layers[0].perceptron[k].synapsys)
                        sum += Math.Pow(j, 2);
                    sum = Math.Pow(sum, 0.5);
                    for (int j = 0; j < _layer.ffann.NumPercept[1]; j++)
                        Layers[0].perceptron[k].synapsys[j] = _layer.b * Layers[0].perceptron[k].synapsys[j] / sum;
                }
            }
        }
    }



    public class Synapsys
    {
        private double[][] Brain;

        public Synapsys(int numNeurons)
        {
            Brain = new double[numNeurons][];
            for (int i = 0; i < numNeurons; i++)
                Brain[i] = new double[numNeurons];
            randomInit();
        }

        public void delNeuron(int[] index)
        {


        }

        public void insNeuron(int[] index)
        {


        }

        public void cutSynapsys(int[] index)
        {


        }

        public void bindNeurons(int[] index)
        {


        }

        private void randomInit()
        {
            Helpers.random(Brain);
        }

    }
}