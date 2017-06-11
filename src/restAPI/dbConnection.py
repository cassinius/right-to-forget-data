import psycopg2
import copy
import json

import datetime
DATE_FORMAT = '%Y%m%d%H%M%S'

DB_NAME = 'iml_anonymization'
DB_HOST = '127.0.0.1'
DB_USER = 'iml_admin'
DB_PASS = 'iml_pass'
DB_TABLE_RAW = 'iml_requests_raw'
DB_TABLE_RESULTS = 'iml_results'

conn = None
cur = None


def connectDB():
    global conn
    global cur
    try:
        conn = psycopg2.connect("dbname=" + DB_NAME + " user=" + DB_USER + " host=" + DB_HOST + " password=" + DB_PASS)
        cur = conn.cursor()
        print( "Successfully connected to DB." )
    except:
        print( "I am unable to connect to the database" )


def getResultsFromDB():
    connectDB()

    try:
        query = """SELECT id, grouptoken, timestamp, target, usertoken FROM %s""" % (DB_TABLE_RESULTS)
        cur.execute(query)
        results = cur.fetchall()
        print( results )
        conn.commit()
        print("Successfully retrieved results.")
        # cur.close()
        conn.close()
        return json.dumps( {"results": results} )
    except:
        print("DB transaction failed... Could not retrieve results!")



def storeRawRequest(request_json, timestamp):
    connectDB()

    db_json = copy.deepcopy(request_json)
    del db_json["csv"]

    try:
        query = """INSERT INTO %s (timestamp, request_raw) VALUES (%s, '%s')""" % (DB_TABLE_RAW, timestamp, json.dumps(db_json))
        cur.execute(query)
        conn.commit()
        print( "Raw request stored successfully." )
        conn.close()
    except:
        print( "DB transaction failed... RAW request was not saved!" )



def storeResult(request, overall_results):
    connectDB()

    try:
        query = """INSERT INTO %s (timestamp, grouptoken, usertoken, target, weights_bias, weights_iml, results_bias, results_iml, plot_url, user_info, survey) VALUES (%s, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')""" % (
            DB_TABLE_RESULTS,
            overall_results['timestamp'],
            overall_results['grouptoken'],
            overall_results['usertoken'],
            overall_results['target'],
            json.dumps(request.json.get('weights').get('bias')),
            json.dumps(request.json.get('weights').get('iml')),
            json.dumps(overall_results['results']['bias']),
            json.dumps(overall_results['results']['iml']),
            overall_results['plotURL'],
            json.dumps(request.json.get('user')),
            json.dumps(request.json.get('survey'))
        )
        cur.execute(query)
        conn.commit()
        print( "Results stored successfully." )
        conn.close()
    except Exception as e:
        if hasattr(e, 'message'):
            print( e.message )
        else:
            print( e )
        print( "DB transaction failed... Results were not saved!" )


if __name__ == "__main__":
    connectDB()