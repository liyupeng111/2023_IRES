
library(R.utils)
library(seqinr)

# Rfam
data_folder <- 'data/rfam/'

Rfam<-read.delim(paste0(data_folder, 'rfam_cis.txt'))

# run once on 10/05/2022

Rfam_url<-'http://http.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/'
for(accession in Rfam$Accession){
  file=paste0(accession, '.fa.gz')
  #download.file(paste0(Rfam_url, file), paste0(data_folder,file))
  #gunzip(paste0(data_folder,file))
}


# get fasta sequences
temp_seq<-NULL
for(accession in Rfam$Accession){
  fa_seq <- read.fasta(file = paste0(data_folder,accession,'.fa'))
  
  seq_names <- names(fa_seq)
  seq_header <- unlist(lapply(fa_seq, function(x){
    getAnnot(x)
  }))
  seq <- unlist(lapply(fa_seq, function(x){
    paste0(getSequence(x),collapse = '')
  }))
  seq_df<-data.frame(Accession=accession, Seq.ID=seq_names, Seq.Header=seq_header, Seq=seq)
  temp_seq<-rbind(temp_seq, seq_df)
}

Rfam_seq<- merge(Rfam, temp_seq)

Rfam_seq$Seq.len<-unlist(lapply(Rfam_seq$Seq, nchar))

Rfam_family_len<-aggregate(Rfam_seq$Seq.len, by=list(Rfam_seq$ID), median)
names(Rfam_family_len)<-c('ID', 'Seq.len.median')

Rfam<-merge(Rfam_family_len, Rfam)
Rfam$IRES<-ifelse(grepl('IRES', Rfam$Type), 1, 0)

Rfam_seq$Seq<-toupper(Rfam_seq$Seq)
Rfam_seq$Seq<-gsub('U', 'T', Rfam_seq$Seq)
Rfam_seq$IRES<-ifelse(grepl('IRES', Rfam_seq$Type), 1, 0)
Rfam_seq<-Rfam_seq[-grep('[^ACGTN]', Rfam_seq$Seq),]

write.csv(Rfam_seq, paste0(data_folder, 'rfam_seq.csv'), row.names = F)

hist(Rfam$Seq.len.median[Rfam$IRES==1])
hist(Rfam$Seq.len.median[Rfam$IRES==0 & Rfam$Seq.len.median>100])

Rfam_seq_100<-Rfam_seq[Rfam_seq$ID %in% Rfam$ID[Rfam$Seq.len.median>100], ]

write.csv(Rfam_seq_100, paste0(data_folder, 'rfam_seq_100.csv'), row.names = F)



