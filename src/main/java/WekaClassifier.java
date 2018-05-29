import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.Stemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.experiment.InstanceQuery;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;


public class WekaClassifier {

    private static Logger LOGGER = Logger.getLogger("WekaClassifier");

    private FilteredClassifier classifier;

    //declare train and test data Instances
    private Instances trainData;


    //declare attributes of Instance
    private ArrayList<Attribute> wekaAttributes;

    //declare and initialize file locations
    private static final String TRAIN_DATA = "src/main/resources/dataset/train.txt";
    private static final String TRAIN_ARFF_ARFF = "src/main/resources/dataset/train.arff";
    private static final String TEST_DATA = "src/main/resources/dataset/test.txt";
    private static final String TEST_DATA_ARFF = "src/main/resources/dataset/test.arff";

    WekaClassifier() {

        /*
         * Class for running an arbitrary classifier on data that has been passed through an arbitrary filter
         * Training data and test instances will be processed by the filter without changing their structure
         */
        classifier = new FilteredClassifier();

        // set Multinomial NaiveBayes as arbitrary classifier
        classifier.setClassifier(new NaiveBayesMultinomial());

        // Declare text attribute to hold the message
        Attribute attributeText = new Attribute("text", (List<String>) null);

        // Declare the label attribute along with its values
        ArrayList<String> classAttributeValues = new ArrayList<>();
        classAttributeValues.add("spam");
        classAttributeValues.add("ham");
        Attribute classAttribute = new Attribute("label", classAttributeValues);

        // Declare the feature vector
        wekaAttributes = new ArrayList<>();
        wekaAttributes.add(classAttribute);
        wekaAttributes.add(attributeText);

    }

    /**
     * load training data and set feature generators
     */
    public void transform(int nGramMinSize, int nGramMaxSize) {
        try {
            trainData = loadRawDataset(TRAIN_DATA);
            saveArff(trainData, TRAIN_ARFF_ARFF);

            // create the filter and set the attribute to be transformed from text into a feature vector (the last one)
            StringToWordVector filter = new StringToWordVector();
            // FIXME: 10.04.18 TEMP
            Stemmer stemmer = new IteratedLovinsStemmer();
            filter.setStemmer(stemmer);
            filter.setAttributeIndices("last");

            //add ngram tokenizer to filter with min and max length set to 1
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(nGramMinSize);
            tokenizer.setNGramMaxSize(nGramMaxSize);
            //use word delimeter
            tokenizer.setDelimiters("\\W");
            filter.setTokenizer(tokenizer);

            //convert tokens to lowercase
            filter.setLowerCaseTokens(true);

            //add filter to classifier
            classifier.setFilter(filter);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * build the classifier with the Training data
     */
    public void fit() {
        try {
            classifier.buildClassifier(trainData);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
        }
    }


    /**
     * classify a new message into spam or ham.
     *
     * @param text to be classified.
     * @return a class label (spam or ham )
     */
    public String predict(String text) {
        try {
            // create new Instance for prediction.
            DenseInstance newinstance = new DenseInstance(2);

            //weka demand a dataset to be set to new Instance
            Instances newDataset = new Instances("predictiondata", wekaAttributes, 1);
            newDataset.setClassIndex(0);

            newinstance.setDataset(newDataset);

            // text attribute value set to value to be predicted
            newinstance.setValue(wekaAttributes.get(1), text);

            // predict most likely class for the instance
            double pred = classifier.classifyInstance(newinstance);

            // return original label
            return newDataset.classAttribute().value((int) pred);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * evaluate the classifier with the Test data
     *
     * @return evaluation summary as string
     */
    public String evaluate() {
        try {
            //load testdata
            Instances testData;
            if (new File(TEST_DATA_ARFF).exists()) {
                testData = loadArff(TEST_DATA_ARFF);
                testData.setClassIndex(0);
            } else {
                testData = loadRawDataset(TEST_DATA);
                saveArff(testData, TEST_DATA_ARFF);
            }

            Evaluation eval = new Evaluation(testData);
//            eval.crossValidateModel(classifier, testData, 2, new Random());
            eval.evaluateModel(classifier, testData);
            double falseNegativeRate = eval.falseNegativeRate(1);
//        if (falseNegativeRate > 0.12) throw new Exception();

            double falsePositiveRate = eval.falsePositiveRate(1);
//        if (falsePositiveRate > 0.12) throw new Exception();

            LOGGER.info("False Negative: Не распознал конкурс: " + falseNegativeRate);
            LOGGER.info("False Positive: Распознал спам как конкурс: " + falsePositiveRate);
            return eval.toSummaryString();
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * This method loads the model to be used as classifier.
     *
     * @param fileName The name of the file that stores the text.
     */
    public void loadModel(String fileName) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            LOGGER.info("Loaded model: " + fileName);
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        } catch (ClassNotFoundException e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * This method saves the trained model into a file. This is done by
     * simple serialization of the classifier object.
     *
     * @param fileName The name of the file that will store the trained model.
     */

    public void saveModel(String fileName) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
            LOGGER.info("Saved model: " + fileName);
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * Loads a dataset in space seperated text file and convert it to Arff format.
     *
     * @param filename The name of the file.
     */
    public Instances loadRawDataset(String filename) {
        /* 
         *  Create an empty training set
         *  name the relation “Rel”.
         *  set intial capacity of 10*
         */

        Instances dataset = new Instances("SMS spam", wekaAttributes, 10);

        // Set class index
        dataset.setClassIndex(0);

        // read text file, parse data and add to instance
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            for (String line;
                (line = br.readLine()) != null;) {
                // split at first occurance of n no. of words
                String[] parts = line.split("\\s+", 2);

                // basic validation
                if (!parts[0].isEmpty() && !parts[1].isEmpty()) {

                    DenseInstance row = new DenseInstance(2);
                    row.setValue(wekaAttributes.get(0), parts[0]);
                    row.setValue(wekaAttributes.get(1), parts[1]);

                    // add row to instances
                    dataset.add(row);
                }

            }

        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        } catch (ArrayIndexOutOfBoundsException e) {
            LOGGER.info("invalid row.");
        }
        return dataset;

    }

    /**
     * Loads a dataset in ARFF format. If the file does not exist, or
     * it has a wrong format, the attribute trainData is null.
     * @param fileName The name of the file that stores the dataset.
     */
    public Instances loadArff(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            ArffReader arff = new ArffReader(reader);
            Instances dataset = arff.getData();
            // replace with logger System.out.println("loaded dataset: " + fileName);
            reader.close();
            return dataset;
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * This method saves a dataset in ARFF format.
     *
     * @param dataset  dataset in arff format
     * @param filename The name of the file that stores the dataset.
     */
    public void saveArff(Instances dataset, String filename) {
        try {
            // initialize
            ArffSaver arffSaverInstance = new ArffSaver();
            arffSaverInstance.setInstances(dataset);
            arffSaverInstance.setFile(new File(filename));
            arffSaverInstance.writeBatch();
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * Main method. With an example usage of this class.
     */
    public static void main(String[] args) {
        final String MODEL = "src/main/resources/models/message.dat";

        WekaClassifier wt = new WekaClassifier();

        if (new File(MODEL).exists()) {
            wt.loadModel(MODEL);
        } else {
            for (int i = 2; i <= 4; i++) {
                wt.transform(i, i);
                wt.fit();
                String evaluate = wt.evaluate();
                LOGGER.info("Min: " + i + " Max: " + i);
                LOGGER.info("Evaluation Result: \n" + evaluate);
                LOGGER.info("text is spam. predicted: " + wt.predict("3 апреля 2018 г, мы разыграем 3 вкусных ужина!!! ДЛЯ участия нужно: 1. Быть жителем г. Новосибирск 2. Быть / Стать участником пабликаhttps://vk.com/big_city_sushi_54 3. нажать \"Мне нравится\" И \"Рассказать Друзьям \" (данный пост) 4. сохранять репост на стене до окончания розыгрыша! Победитель будет выбран генератором случайных пользователей. Так же вы можете заказать наши сеты по акции \"32 кусочка\" - https://vk.cc/6YBscl и \"2 кг за 850 рублей\" - https://vk.cc/6M1O8V"));
                LOGGER.info("text is spam. predicted: " + wt.predict("Завтра 2 апреля мы разыграемСЕРТИФИКАТ на 1500 рублей на покупку в группе спонсора https://m.vk.com/wall-116141977_45824 апреля ОДНА ИГРУШКА НА ВЫБОР ПОБЕДИТЕЛЯ https://vk.com/wall-116141977_45944 апреля КОСМЕТИКА ДЛЯ ВОЛОС один товар на выбор https://vk.com/wall-116141977_4597ПЕРЕХОДИТЕ ПО ССЫЛОЧКАМ И ДЕЛАЙТЕ РЕПОСТЫ"));
                LOGGER.info("text is spam. predicted: " + wt.predict("1 апреля / Smoking / Нам 1 год, День Рождения Глаза Геометрии : [id9888443|@annaglushchenko]ВНИМАНИЕ!!!\uD83D\uDC4F\uD83D\uDC4F\uD83D\uDC4F ДЕНЬ РОЖДЕНИЯ @smoking_mur Нам 1 год \uD83D\uDC90\uD83D\uDC90\uD83D\uDC90\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDE09\uD83D\uDE09\uD83D\uDE09 Мы хотим сказать спасибо нашим дорогим гостям за этот год, за то, что были с нами! Спасибо тем, кто вступил в семью @smoking_mur вас стало очень много, мы всех знаем по именам и очень приятно видеть когда люди после первого посещения возвращаются к нам снова \uD83D\uDE18спасибо вам!!!! \uD83D\uDC90\uD83D\uDC90\uD83D\uDC90 А наши постоянные клиенты, которые на протяжении многих лет ходят в наше заведение - они все знают без слов \uD83D\uDE09 Что мы их любим \uD83D\uDE18\uD83C\uDF37 Хотим пригласить всех 1 АПРЕЛЯ в 19:00 на празднование нашего ДНЯ РОЖДЕНИЯ! За хорошее настроение и смех будет отвечать наш незаменимый @dmitryyavkin за фоторепортаж наша любимая @anna_glushchenko_ \uD83D\uDCF8Угостим каждого бокалом игристого \uD83C\uDF7E\uD83C\uDF7E\uD83C\uDF7E\uD83E\uDD17\uD83E\uDD17\uD83E\uDD17Разыграем сертификат номиналом 1️⃣5️⃣0️⃣0️⃣ рублей среди наших гостей\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDC4D Всех ждём! Будет весело \uD83D\uDE24\uD83D\uDE18\uD83D\uDCAD\uD83D\uDE1C☎️Бронь столов 78-29-65#smoking_mur #smoking #lounge #murmansk #кальянвмурманске #hookah #darkside #wto #смокингмурманск #мурманск"));
                LOGGER.info("text is ham. predicted: " + wt.predict("ВНИМАНИЕ РОЗЫГРЫШ  Разыгрывается 200 рублей на баланс мобильного телефона Для участия необходимо: ✅ Быть участником нашей группы☝ ✅ Сделать репост этой записи себе на стену Розыгрыш состоится 10 мая с помощью приложения ВК #Халява #Розыгрыш #Бесплатно#Череповец#Набаланс \uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCF1 ✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81\uD83D\uDD28✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81✨\uD83C\uDF81\uD83D\uDE9A\uD83D\uDEAA\uD83D\uDD28  Акция !!!Самые низкие цены на весь модельный ряд! \uD83C\uDF81 Приятный подарок каждому покупателю-доставка по городу Череповцу,подъем на этаж,демонтаж дверного блока и установка входной двери!!! Акция действует при покупке входных дверей от 11500 рублей!!! ☎281362 или \uD83D\uDCF189211303091"));

            }
        }

//            wt.saveModel(MODEL);
    }
//
//        //run few predictions
//        LOGGER.info("text is spam. predicted: " + wt.predict("3 апреля 2018 г, мы разыграем 3 вкусных ужина!!! ДЛЯ участия нужно: 1. Быть жителем г. Новосибирск 2. Быть / Стать участником пабликаhttps://vk.com/big_city_sushi_54 3. нажать \"Мне нравится\" И \"Рассказать Друзьям \" (данный пост) 4. сохранять репост на стене до окончания розыгрыша! Победитель будет выбран генератором случайных пользователей. Так же вы можете заказать наши сеты по акции \"32 кусочка\" - https://vk.cc/6YBscl и \"2 кг за 850 рублей\" - https://vk.cc/6M1O8V"));
//        LOGGER.info("text is spam. predicted: " + wt.predict("Завтра 2 апреля мы разыграемСЕРТИФИКАТ на 1500 рублей на покупку в группе спонсора https://m.vk.com/wall-116141977_45824 апреля ОДНА ИГРУШКА НА ВЫБОР ПОБЕДИТЕЛЯ https://vk.com/wall-116141977_45944 апреля КОСМЕТИКА ДЛЯ ВОЛОС один товар на выбор https://vk.com/wall-116141977_4597ПЕРЕХОДИТЕ ПО ССЫЛОЧКАМ И ДЕЛАЙТЕ РЕПОСТЫ"));
//        LOGGER.info("text is spam. predicted: " + wt.predict("1 апреля / Smoking / Нам 1 год, День Рождения Глаза Геометрии : [id9888443|@annaglushchenko]ВНИМАНИЕ!!!\uD83D\uDC4F\uD83D\uDC4F\uD83D\uDC4F ДЕНЬ РОЖДЕНИЯ @smoking_mur Нам 1 год \uD83D\uDC90\uD83D\uDC90\uD83D\uDC90\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDE09\uD83D\uDE09\uD83D\uDE09 Мы хотим сказать спасибо нашим дорогим гостям за этот год, за то, что были с нами! Спасибо тем, кто вступил в семью @smoking_mur вас стало очень много, мы всех знаем по именам и очень приятно видеть когда люди после первого посещения возвращаются к нам снова \uD83D\uDE18спасибо вам!!!! \uD83D\uDC90\uD83D\uDC90\uD83D\uDC90 А наши постоянные клиенты, которые на протяжении многих лет ходят в наше заведение - они все знают без слов \uD83D\uDE09 Что мы их любим \uD83D\uDE18\uD83C\uDF37 Хотим пригласить всех 1 АПРЕЛЯ в 19:00 на празднование нашего ДНЯ РОЖДЕНИЯ! За хорошее настроение и смех будет отвечать наш незаменимый @dmitryyavkin за фоторепортаж наша любимая @anna_glushchenko_ \uD83D\uDCF8Угостим каждого бокалом игристого \uD83C\uDF7E\uD83C\uDF7E\uD83C\uDF7E\uD83E\uDD17\uD83E\uDD17\uD83E\uDD17Разыграем сертификат номиналом 1️⃣5️⃣0️⃣0️⃣ рублей среди наших гостей\uD83D\uDC4D\uD83D\uDC4D\uD83D\uDC4D Всех ждём! Будет весело \uD83D\uDE24\uD83D\uDE18\uD83D\uDCAD\uD83D\uDE1C☎️Бронь столов 78-29-65#smoking_mur #smoking #lounge #murmansk #кальянвмурманске #hookah #darkside #wto #смокингмурманск #мурманск"));
//        LOGGER.info("text is ham. predicted: " + wt.predict("ВНИМАНИЕ РОЗЫГРЫШ  Разыгрывается 200 рублей на баланс мобильного телефона Для участия необходимо: ✅ Быть участником нашей группы☝ ✅ Сделать репост этой записи себе на стену Розыгрыш состоится 10 мая с помощью приложения ВК #Халява #Розыгрыш #Бесплатно#Череповец#Набаланс \uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCB8\uD83D\uDCF1 ✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81\uD83D\uDD28✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81✨\uD83D\uDEAA\uD83D\uDE9A\uD83C\uDF81✨\uD83C\uDF81\uD83D\uDE9A\uD83D\uDEAA\uD83D\uDD28  Акция !!!Самые низкие цены на весь модельный ряд! \uD83C\uDF81 Приятный подарок каждому покупателю-доставка по городу Череповцу,подъем на этаж,демонтаж дверного блока и установка входной двери!!! Акция действует при покупке входных дверей от 11500 рублей!!! ☎281362 или \uD83D\uDCF189211303091"));
//
//        //run evaluation
//        LOGGER.info("Evaluation Result: \n" + wt.evaluate());
}