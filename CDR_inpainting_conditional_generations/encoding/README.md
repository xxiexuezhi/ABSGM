
To generate conditional samples, you have to first enocde epitope-antibody complex using the encoding file, then provide the encoded matrix incuding H1,H2, or H3 the regions to be masked during inference. 
For convenience, I made a pandas data frame stored all the related antigen information called . 

1. For quick encoding, please refer to encoding_reurn_read_from_df_epitopes.ipynb  I have pre-compute and saved the epitope reidues a processsed_df_table_with_epitope_residue_info_2.pkl. The link is here: https://drive.google.com/file/d/1NXjaV8l0m99IDLYjM-cY3XMpRZ8l1n0_/view?usp=sharing
 It would read the saved epitotpe info and combine with the HVV to get the encoded file.

3. To generate the epitope information, please check the jupyter notebook
  
5. Conditional generation additionally requires the epitope-antibody complex with the designed CDR been masked out.
Please noted, this masked information should be stored indside matrix as the "ss_helices: ".  For instance, to mask and re-generated H3 regions, you can run the following command:






