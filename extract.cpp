#include "TTree.h"
#include "TFile.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>


using namespace std;


void print_usage() {
    cout << "Usage:\n"
            "  extract <input.root> [output.csv]" << endl;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "Too few parameters, please specify ROOT file path.\n";
        print_usage();
        return 1;
    }


    {
        struct stat buffer;   
        if ( stat(argv[1], &buffer) != 0 ) {
            cout << "Error: cannot open file '" << argv[1] << "', check path and permissions.\n";
            return 1;
        }
    }


    string outputFileName;
    if (argc >= 3)
        outputFileName = argv[2];
    else {
        outputFileName = argv[1];
        auto pos = outputFileName.find_last_of('.');
        if (pos == string::npos) 
            outputFileName += ".csv";
        else {
            auto slashPos = outputFileName.find_last_of('/');
            if (slashPos == string::npos)
                slashPos = 0;
            else
                slashPos ++;
            outputFileName = outputFileName.substr(slashPos, pos - slashPos) + ".csv";
        }
    }


    TFile *f = TFile::Open(argv[1], "r");
    TTree *t = dynamic_cast<TTree*>(f->Get("Vars"));

    Float_t Lepton_PX, Lepton_PY, Lepton_PZ, Lepton_E;
    Float_t MET_PX, MET_PY, MET_PZ, MET_E;
    Float_t MtW, MET, Pt_Lep, DPhi_LepNu;

    t->SetBranchAddress("Lepton_PX", &Lepton_PX);
    t->SetBranchAddress("Lepton_PY", &Lepton_PY);
    t->SetBranchAddress("Lepton_PZ", &Lepton_PZ);
    t->SetBranchAddress("Lepton_E" , &Lepton_E );
    t->SetBranchAddress("MET_PX", &MET_PX);
    t->SetBranchAddress("MET_PY", &MET_PY);
    t->SetBranchAddress("MET_PZ", &MET_PZ);
    t->SetBranchAddress("MET_E", &MET_E );
    t->SetBranchAddress("MtW", &MtW);
    t->SetBranchAddress("MET", &MET);
    t->SetBranchAddress("Pt_Lep", &Pt_Lep);
    t->SetBranchAddress("DPhi_LepNu", &DPhi_LepNu);

    ofstream of(outputFileName, ios_base::out | ios_base::trunc);
    of << "MtW, MET, Pt_Lep, DPhi_LepNu, Lepton_PX, Lepton_PY, Lepton_PZ, Lepton_E, MET_PX, MET_PY, MET_E\n";

    for (size_t i = 0; i < t->GetEntries(); i++) {
        t->GetEntry(i);
        of << MtW << ", " << MET << ", " << Pt_Lep << ", " << DPhi_LepNu << ", " << Lepton_PX << ", " << Lepton_PY << ", " << Lepton_PZ << ", " << Lepton_E << ", " << MET_PX << ", " << MET_PY << ", " << MET_E << '\n';
    }

    of.close();
    cout << " input: " << argv[1] << endl;
    cout << "output: " << outputFileName << endl;

    cout << t->GetEntries() << endl;

    return 0;
}
