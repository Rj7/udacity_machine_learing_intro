#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
import math
sys.path.append("../final_project")

def poiEmails():
    email_list = ["kenneth_lay@enron.net",
            "kenneth_lay@enron.com",
            "klay.enron@enron.com",
            "kenneth.lay@enron.com",
            "klay@enron.com",
            "layk@enron.com",
            "chairman.ken@enron.com",
            "jeffreyskilling@yahoo.com",
            "jeff_skilling@enron.com",
            "jskilling@enron.com",
            "effrey.skilling@enron.com",
            "skilling@enron.com",
            "jeffrey.k.skilling@enron.com",
            "jeff.skilling@enron.com",
            "kevin_a_howard.enronxgate.enron@enron.net",
            "kevin.howard@enron.com",
            "kevin.howard@enron.net",
            "kevin.howard@gcm.com",
            "michael.krautz@enron.com"
            "scott.yeager@enron.com",
            "syeager@fyi-net.com",
            "scott_yeager@enron.net",
            "syeager@flash.net",
            "joe'.'hirko@enron.com",
            "joe.hirko@enron.com",
            "rex.shelby@enron.com",
            "rex.shelby@enron.nt",
            "rex_shelby@enron.net",
            "jbrown@enron.com",
            "james.brown@enron.com",
            "rick.causey@enron.com",
            "richard.causey@enron.com",
            "rcausey@enron.com",
            "calger@enron.com",
            "chris.calger@enron.com",
            "christopher.calger@enron.com",
            "ccalger@enron.com",
            "tim_despain.enronxgate.enron@enron.net",
            "tim.despain@enron.com",
            "kevin_hannon@enron.com",
            "kevin'.'hannon@enron.com",
            "kevin_hannon@enron.net",
            "kevin.hannon@enron.com",
            "mkoenig@enron.com",
            "mark.koenig@enron.com",
            "m..forney@enron.com",
            "ken'.'rice@enron.com",
            "ken.rice@enron.com",
            "ken_rice@enron.com",
            "ken_rice@enron.net",
            "paula.rieker@enron.com",
            "prieker@enron.com",
            "andrew.fastow@enron.com",
            "lfastow@pdq.net",
            "andrew.s.fastow@enron.com",
            "lfastow@pop.pdq.net",
            "andy.fastow@enron.com",
            "david.w.delainey@enron.com",
            "delainey.dave@enron.com",
            "'delainey@enron.com",
            "david.delainey@enron.com",
            "'david.delainey'@enron.com",
            "dave.delainey@enron.com",
            "delainey'.'david@enron.com",
            "ben.glisan@enron.com",
            "bglisan@enron.com",
            "ben_f_glisan@enron.com",
            "ben'.'glisan@enron.com",
            "jeff.richter@enron.com",
            "jrichter@nwlink.com",
            "lawrencelawyer@aol.com",
            "lawyer'.'larry@enron.com",
            "larry_lawyer@enron.com",
            "llawyer@enron.com",
            "larry.lawyer@enron.com",
            "lawrence.lawyer@enron.com",
            "tbelden@enron.com",
            "tim.belden@enron.com",
            "tim_belden@pgn.com",
            "tbelden@ect.enron.com",
            "michael.kopper@enron.com",
            "dave.duncan@enron.com",
            "dave.duncan@cipco.org",
            "duncan.dave@enron.com",
            "ray.bowen@enron.com",
            "raymond.bowen@enron.com",
            "'bowen@enron.com",
            "wes.colwell@enron.com",
            "dan.boyle@enron.com",
            "cloehr@enron.com",
            "chris.loehr@enron.com"
        ]
    return email_list


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_email_list = poiEmails()

poi_dictionary = {}

count = 0
count1 = 0
for key, value in enron_data.items():
    if value["email_address"] in poi_email_list and value["total_payments"] != "NaN" :
        poi_dictionary[key] = value
        count += 1
        print key
    if value["total_payments"] != "NaN":
        count1 += 1

print count1
print count
print len(poi_dictionary)
print len(enron_data)
print enron_data["LAY KENNETH L"]



