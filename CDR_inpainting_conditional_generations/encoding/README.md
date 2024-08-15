
To generate conditional samples, first, encode the epitope-antibody complex using the encoding file, then provide the encoded matrix, including H1, H2, or H3, the regions to be masked during inference. A pandas data frame that stores the related antigen information examples was created for convenience.

For convenience, I made a pandas data frame stored all the related antigen information called . 

1. For quick encoding, please refer to encoding_reurn_read_from_df_epitopes.ipynb jupyter notebook, where the example epitope residues were pre-computed and saved as processsed_df_table_with_epitope_residue_info_2.pkl. [The link](https://drive.google.com/file/d/1NXjaV8l0m99IDLYjM-cY3XMpRZ8l1n0_/view?usp=sharing). It would read and combine the saved epitope info with the HVV to get the encoded file.

2. To generate the epitope information, please check the jupyter notebook.

  
3. Conditional generation requires the epitope-antibody complex with the designed CDR to be masked. This masked information should be stored inside the matrix as the "ss_helices". 







