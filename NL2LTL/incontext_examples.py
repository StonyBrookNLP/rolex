import numpy as np
import pandas as pd
import random

incontext = """NL: The sr_slotid result MUST equal sa_slotid .
PK:
LTL: ALWAYS ( get_result(sr_slotid) == sa_slotid )
NL:  However, if gdia_maxcount is zero , NFS4ERR_TOOSMALL MUST NOT be returned .
PK: A is returned=>return ( E ) WITH E . A
LTL: ALWAYS ( ( gdia_maxcount==0 ) -> NOT ( return ( E ) WITH E . NFS4ERR_TOOSMALL ) )
NL: If the source offset or the source offset plus count is greater than the size of the source file , the operation MUST fail with NFS4ERR_INVAL .
PK: fail with A=>fail ( E ) WITH E . A
LTL: ALWAYS ( ( ( get_offset(source)>get_size(cfh)) OR (get_offset(source)+count>get_size(cfh) ) ) -> ( fail ( E ) WITH E . NFS4ERR_INVAL ) )
NL: If it does not want to support a hole , it MUST use READ .
PK: it=>server<SEP>support A=>support ( E )  WITH E . A<SEP>read=>READ ( E )
LTL: ALWAYS ( NOT ( support ( E ) WITh E . server) -> ( READ ( E ) ) )
NL: The array contents MUST be contiguous in the file .
PK: file=>cfh<SEP>array=>arr<SEP>A is contiguous=>contiguous ( E ) WITH E . A
LTL: ALWAYS ( contiguous ( E ) WITH E . get_contents(arr) WITH E . cfh )
NL: If a previous filehandle was saved , then it is no longer accessible .
PK: previous filehandle=>prev_cfh<SEP>A is saved => save ( E ) WITH E . A<SEP>A is accessible => access ( E ) WITH E . A
LTL: ALWAYS ( ( save ( E ) WITH E . prev_cfh ) -> NOT ( access ( E ) WITH E . prev_cfh ) )
NL: If none exist , then NFS4ERR_NOENT will be returned .
PK: A exists=>exist ( E ) WITH E . A<SEP>A is returned=>return ( E ) WITH E . A<SEP>none => cfh
LTL: ALWAYS ( NOT ( exist ( E ) WITH E . cfh) -> ( return ( E ) WITH E . NFS4ERR_NOENT ) )
NL: If the client specified CDFC4_FORE , the server MUST return CDFS4_FORE .
PK: specify A=>specify ( E ) WITH E . A<SEP>return A=>return ( E ) WITH E . A
LTL: ALWAYS ( ( specify ( E ) WITH E . CDFC4_FORE ) -> ( return ( E ) WITH E . CDFS4_FORE ) )"""




if __name__ == "__main__":
    x = incontext.splitlines()
    x = [i[3:] for i in x]
    x = [i.strip() for i in x]

    print(x)

    nl = [x[i] for i in range(0,len(x),3)] 
    nl = [i.strip() for i in nl]
    fl = [x[i+2] for i in range(0,len(x),3)] 
    fl = [i.strip() for i in fl]
    pk_gold = [x[i+1] for i in range(0,len(x),3)]
    pk_gold = [i.strip() for i in pk_gold]
    pk_gold = [i.split("|") for i in pk_gold]
    print(pk_gold)
    for i in range(0,len(pk_gold)):
        for j in range(0,len(pk_gold[i])):
            pk_gold[i][j] = pk_gold[i][j].strip()
    pk_gold = ["<SEP>".join(i) for i in pk_gold]

    df = pd.DataFrame({"nl":nl,"fl":fl,"pk_gold":pk_gold})
    df.to_csv("dataset/incontext_examples.csv")