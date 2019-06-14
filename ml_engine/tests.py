from ml_engine.models.text_classification.intent_classification import IntentClassification


def test_model():
    print('Testing model')

    classifier = IntentClassification([
        {'sentence': 'can you help me in getting an order for moto g5', 'class': 'place_order'},
        {'sentence': 'can you place an order for samsung j3', 'class': 'place_order'},
        {'sentence': 'please help me in buying apple iphone', 'class': 'place_order'},
        {'sentence': 'i want to buy a dell vostro', 'class': 'place_order'},

        # {'sentence': "please let me know the details of laptop", 'class': 'fetch_product'},
        # {'sentence': 'can you  fetch detail of dell laptop', 'class': 'fetch_product'},
        # {'sentence': 'Can you provide the details', 'class': 'fetch_product'},
        # {'sentence': "Please show some products.", 'class': 'fetch_product'},
        # {'sentence': 'can i get the details of smart phone', 'class': 'fetch_product'},
        # {'sentence': 'please help me with the details of  mobile phone', 'class': 'fetch_product'},
        # {'sentence': 'i want to know the price of laptop', 'class': 'fetch_product'},

        {'sentence': 'Any cab nearby location', 'class': 'cab_category_details'},
        {'sentence': 'i want to book to ola cab', 'class': 'cab_category_details'},
        {'sentence': 'Need a cab as fast as possible', 'class': 'cab_category_details'},
        {'sentence': 'need to get a cab', 'class': 'cab_category_details'},
        {'sentence': 'ola  cab required at my location', 'class': 'cab_category_details'},

        {'sentence': 'Need to book an auto with  for tomorrow to a city', 'class': 'ola_auto_details'},
        {'sentence': 'gives information about  auto', 'class': 'ola_auto_details'},
        {'sentence': 'can you book  auto cab', 'class': 'ola_auto_details'},
        {'sentence': 'auto booking is too hectic or cumbersome,so need to get the booking in very less time',
         'class': 'ola_auto_details'},
        {'sentence': 'want to book an  auto with cheapest fare', 'class': 'ola_auto_details'},
        {'sentence': 'book an ola auto for me', 'class': 'ola_auto_details'},

        {'sentence': 'Gives information or details of ola share', 'class': 'ola_share_details'},
        {'sentence': 'Can you book a ola share cab', 'class': 'ola_share_details'},
        {'sentence': 'share booking is convenient for me, can you book one for me', 'class': 'ola_share_details'},
        {'sentence': 'ola share with two people in 10 minutes', 'class': 'ola_share_details'},
        {'sentence': 'looking to book an ola share with cheapest price to noida city centre',
         'class': 'ola_share_details'},
        {'sentence': 'want to book a share cab', 'class': 'ola_share_details'},

        {'sentence': 'details of cab available at location', 'class': 'cab_fetch_service'},
        {'sentence': 'show me detail of cab', 'class': 'cab_fetch_service'},
        {'sentence': 'need a cab', 'class': 'cab_fetch_service'},
        {'sentence': 'give me the details of cab available near me', 'class': 'cab_fetch_service'},
        {'sentence': 'which are the cabs available near me', 'class': 'cab_fetch_service'},
        {'sentence': 'Please show some cabs', 'class': 'cab_fetch_service'},

        {"class": "greeting", "sentence": "how are you?"},
        {"class": "greeting", "sentence": "how is your day?"},
        {"class": "greeting", "sentence": "good day"},
        {"class": "greeting", "sentence": "how is it going today?"},

        {"class": "goodbye", "sentence": "have a nice day"},
        {"class": "goodbye", "sentence": "see you later"},
        {"class": "goodbye", "sentence": "good night"},
        {"class": "goodbye", "sentence": "talk to you soon"},
        {"class": "goodbye", "sentence": "farewell"},
        {"class": "goodbye", "sentence": "need to go,bye"},

        {"class": "sandwich", "sentence": "make me a sandwich"},
        {"class": "sandwich", "sentence": "can you make a sandwich?"},
        {"class": "sandwich", "sentence": "having a sandwich today?"},
        {"class": "sandwich", "sentence": "what's for lunch?"},

        {"class": "sandwich", "sentence": "what's cylinder"},

        {"class": "river_bank", "sentence": "I want to visit river bank"},
        {"class":"river_bank","sentence":"river bank is a nice place to have fun"},

        {"class": "bank", "sentence": "I want to deposit money in bank"},
        {"class": "bank", "sentence": "I want to visit bank for depositing money"},
        {"class": "bank", "sentence": "need to visit bank for money depositing"},
        {"class": "bank", "sentence": "going to bank for loan"},
        {"class": "bank", "sentence": "I want to deposit money in bank"},


    ])
    # classifier = IntentClassification(
    #    [{'sentence': 'apply for leaves', 'class': 'leaves'}, {'sentence': 'details of leaves', 'class': 'details'}])
    classifier.configure_classifier()
    '''
    print("testing score:%s" % classifier.model_testing_accuracy(
        [
            {"sentence": "I want to place an order for homemade biscuits", "class": 'place_order'},
            {"sentence": 'please show some nike shoes.', "class": 'fetch_product'},
            {"sentence": "I want to buy t-shirt", "class": 'place_order'},

            {"sentence": "all right then", "class": 'goodbye'},
            {"sentence": "talk to you later", "class": 'goodbye'},
            {"sentence": "you too,bye", "class": 'goodbye'},
            {"sentence": "take care", "class": 'goodbye'},
            {"sentence": "I have to go,good bye", "class": 'goodbye'},

            {"sentence": "feeling hungry,please make a sandwich for me", "class": "sandwich"},
            {"sentence": "lets go for lunch", "class": "sandwich"},
            {"sentence": "I need to go for dinner and eat some good food.", "class": "sandwich"},

            {"sentence": "I need a cab to go hometown", "class": "cab_category_details"},
            {"sentence": "show cabs near me", "class": "cab_fetch_service"},

            {"sentence": "Please book a ola share for me", "class": "ola_share_details"},
            {"sentence": "book a ola auto for me", "class": "ola_auto_details"},
            {"sentence": "ola auto is so expensive, book some cheapest cab", "class": "cab_category_details"},
            {"sentence": "book cab which comes as fast as possible", "class": "cab_category_details"},

            {"sentence": "please share some cab prices", "class": "cab_category_details"},
            {"sentence": "apply for leaves", "class": "cab_category_details"},
            {"sentence": "fetch my leaves details", "class": "cab_fetch_service"},
            {"sentence": "let me know what you have made lunch and dinner?", "class": "sandwich"},
            {"sentence": "its hard to find cab but for now i want to see some food options", "class": "sandwich"},
            {"sentence": "find some good products for tshirts", "class": "fetch_product"},

        ]))'''

    print("training accuracy %s" % classifier.model_training_accuracy())
    # print(classifier.classify(['I want to buy and know the details of dell vostro']))
    print(classifier.classify(
        ["this weekend going to visit river bank"]))

    # {"sentence": "I need to go for dinner and eat some good food.", "class": "sandwich"}
    # {"sentence": "I need a cab to go hometown", "class": "cab_category_details"},
    # {"sentence": "ola auto is so expensive, book some cheapest cab", "class": "cab_category_details"},
    # {"sentence": "please share some cab prices", "class": "cab_category_details"},
    # {"sentence": "apply for leaves", "class": "cab_category_details"},
    # {"sentence": "fetch my leaves details", "class": "cab_fetch_service"},
    # {"sentence": "let me know what you have made lunch and dinner?", "class": "sandwich"}
    # {"sentence": "its hard to find cab but for now i want to see some food options", "class": "sandwich"},
