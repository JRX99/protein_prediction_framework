from transformers import AutoTokenizer, EsmForProteinFolding
import csv,random,re,os
import numpy as np
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from scipy.special import softmax
def generate_random_protein_sequence(min_length, max_length):
    """
    Generate a random protein sequence with a variable length between min_length and max_length.

    Args:
    min_length (int): Minimum length of the protein sequence.
    max_length (int): Maximum length of the protein sequence.

    Returns:
    str: Randomly generated protein sequence.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    sequence_length = random.randint(min_length, max_length)
    random_sequence = [random.choice(amino_acids) for _ in range(sequence_length)]
    return "".join(random_sequence)

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def convert_outputs_to_pae(output):
	pae = (output["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(-1) * 31
	mask = output["atom37_atom_exists"][0,:,1] == 1
	mask = mask.cpu()
	pae = pae[mask,:][:,mask]       
	return pae




tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("/storage/plzen4-ntis/home/jrx99/esmfold/", low_cpu_mem_usage=True)

model = model.cuda()
model.esm = model.esm.half()
print("model loaded!")
torch.backends.cuda.matmul.allow_tf32 = True









# Number of rows you want in the CSV file
batch_size=1000000000

for b in range(batch_size):
  
  luciferase="KPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPGFKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIKGAGGD"
  linker1="GSSGSSGSSGSSGSSGSS"
  linker2="GGGGGGGGGGGGGGGGGG"
  # Create a CSV file and write data to it

  generated_sequence=generate_random_protein_sequence(15,20)
  sequences=[generated_sequence+linker1+luciferase,luciferase+linker1+generated_sequence,generated_sequence+linker2+luciferase,luciferase+linker2+generated_sequence]
  jobnames=["GSS_linker_start","GSS_linker_end","G_linker_start","G_linker_end",]
  for i in range (4):




    jobname = jobnames[i]
    jobname = re.sub(r'\W+', '', jobname)[:50]
    
    sequence = sequences[i]
    num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
    
    ID = generated_sequence+"_"+jobname
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length",length)
    
    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"
    
    torch.cuda.empty_cache()
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()
    # optimized for Tesla T4
    

    
    with torch.no_grad():
        output = model(tokenized_input)
    
    pdb = convert_outputs_to_pdb(output)
    pae = convert_outputs_to_pae(output)
    print(ID)
    output_directory = f"/storage/plzen4-ntis/home/jrx99/output_esm/{ID}"
    print("Output Directory:", output_directory)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Change permissions to allow writing
    os.system(f"chmod a+w {output_directory}")

    # Define the directory path
    directory_path = output_directory
    
    # Saving files
    np.savetxt(f"{directory_path}/{ID}.pae.txt", pae, "%.3f")
    for idx, pdb_string in enumerate(pdb):
        with open(f"{directory_path}/{ID}_{idx}.pdb", "w") as out:
            out.write(pdb_string)