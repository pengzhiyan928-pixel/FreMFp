# FreMFp
Machine learning with frequency molecular fingerprint (FreMFp) for OSCs donor materials
Based on the design process of polymer donor materials, we proposed the frequency molecular fingerprint (FreMFp).
The main tasks in this project include:
1.Input file (test.csv) and run brics_idmap.py to get the molecular substructures obtained by BRICS rule-based fragmentation and produces a fragment list (batchsplit2.CSV) file, with one row per    molecule and its fragments. 
2.Input the Step-1 fragmentation result (batchsplit2.CSV) file and run post_brics.py to get the global fragment vocabulary & ID map, exporting fragment_total_list.csv (frag_id, fragment_smiles),    index_data.csv (per-molecule fragment-ID matrix), and frags_exploded.csv (long format).
3.Using the fragment index from the previous step ( index_data.csv) and the fragment vocabulary (fragment_total_list.csv), run the make_fingerprints.py script to generate per-molecule frequency     fragment fingerprints: counts.csv


  
  
 
