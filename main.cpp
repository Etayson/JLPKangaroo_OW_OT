/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Kangaroo.h"
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("Kangaroo [-v] [-t nbThread] [-d dpBit] [gpu] [-check]\n");
  printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
  printf("         inFile\n");
  printf(" -v: Print version\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
  printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
  printf(" -t nbThread: Secify number of thread\n");
  printf(" -w workfile: Specify file to save work into (current processed key only)\n");
  printf(" -i workfile: Specify file to load work from (current processed key only)\n");
  printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
  printf(" -ws: Save kangaroos in the work file\n");
  printf(" -wsplit: Split work file of server and reset hashtable\n");
  printf(" -wm file1 file2 destfile: Merge work file\n");
  printf(" -wt timeout: Save work timeout in millisec (default is 3000ms)\n");
  printf(" -winfo file1: Work file info file\n");
  printf(" -m maxStep: number of operations before give up the search (maxStep*expected operation)\n");
  printf(" -s: Start in server mode\n");
  printf(" -c server_ip: Start in client mode and connect to server server_ip\n");
  printf(" -sp port: Server port, default is 17403\n");
  printf(" -nt timeout: Network timeout in millisec (default is 3000ms)\n");
  printf(" -o fileName: output result to fileName\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check GPU kernel vs CPU\n");
  printf(" inFile: intput configuration file\n");
  exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

double getDouble(string name,char *v) {

  double r;

  try {

    r = std::stod(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens,const string &text,char sep) {

  size_t start = 0,end = 0;
  tokens.clear();
  int item;

  try {

    while((end = text.find(sep,start)) != string::npos) {
      item = std::stoi(text.substr(start,end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  }
  catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}
// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
static vector<int> gpuId = { 0 };
static vector<int> gridSize;
static string workFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = 60;
static bool saveKangaroo = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = 3000;
static int ntimeout = 3000;
static int port = 17403;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;
static string Gx ="79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static string Gy ="483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
static string multG = "1";
static string prejmpbits = "0";
static string multiplerJMPstr = "1";

int main(int argc, char* argv[]) {

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  

  int a = 1;
  nbCPUThread = Timer::getCoreNumber();

#ifdef USE_SYMMETRY
  printf("Kangaroo v" RELEASE " (with symmetry)\n");
#else
#ifdef USE_YNEG
  printf("Kangaroo v" RELEASE " (with Yneg)\n");
#else
#ifdef ONLY_WILD
  printf("Kangaroo v" RELEASE " (Only Wild)\n");
#else
#ifdef ONLY_TAME
  printf("Kangaroo v" RELEASE " (Only Tame)\n");
#else
  printf("Kangaroo v" RELEASE "\n");
#endif  
#endif
#endif
  
  
#endif

  while (a < argc) {

    if(strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-d") == 0) {
      a++;
      dp = getInt("dpSize",argv[a]);
      a++;
    } else if (strcmp(argv[a], "-h") == 0) {
      printUsage();
    } else if(strcmp(argv[a],"-l") == 0) {

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if(strcmp(argv[a],"-w") == 0) {
      a++;
      workFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-i") == 0) {
      a++;
      iWorkFile = string(argv[a]);
      a++;
    }  else if(strcmp(argv[a],"-wm") == 0) {
      a++;
      merge1 = string(argv[a]);
      a++;
      merge2 = string(argv[a]);
      a++;
      mergeDest = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-winfo") == 0) {
      a++;
      infoFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-o") == 0) {
      a++;
      outputFile = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wi") == 0) {
      a++;
      savePeriod = getInt("savePeriod",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-wt") == 0) {
      a++;
      wtimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-nt") == 0) {
      a++;
      ntimeout = getInt("timeout",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-m") == 0) {
      a++;
      maxStep = getDouble("maxStep",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-ws") == 0) {
      a++;
      saveKangaroo = true;
    } else if(strcmp(argv[a],"-wsplit") == 0) {
      a++;
      splitWorkFile = true;
    } else if(strcmp(argv[a],"-s") == 0) {
      a++;
      serverMode = true;
    } else if(strcmp(argv[a],"-c") == 0) {
      a++;
      serverIP = string(argv[a]);
      a++;
    } else if(strcmp(argv[a],"-sp") == 0) {
      a++;
      port = getInt("serverPort",argv[a]);
      a++;
    } else if(strcmp(argv[a],"-gpu") == 0) {
      gpuEnable = true;
      a++;
    } else if(strcmp(argv[a],"-gpuId") == 0) {
      a++;
      getInts("gpuId",gpuId,string(argv[a]),',');
      a++;
    } else if(strcmp(argv[a],"-g") == 0) {
      a++;
      getInts("gridSize",gridSize,string(argv[a]),',');
      a++;
    } else if(strcmp(argv[a],"-v") == 0) {
      ::exit(0);
    } else if(strcmp(argv[a],"-check") == 0) {
      checkFlag = true;
      a++;
    }else if (strcmp(argv[a], "-gx") == 0) {
        a++;
        Gx = string(argv[a]);
        printf("New Gx %s \n", argv[a]);
        a++;
    }else if (strcmp(argv[a], "-gy") == 0) {
        a++;
        Gy = string(argv[a]);
        printf("New Gy %s \n", argv[a]);
        a++;
    }else if (strcmp(argv[a], "-gmult") == 0) {
        a++;
        multG = string(argv[a]);
        printf("New Gmult %s \n", argv[a]);
        a++;
    }else if (strcmp(argv[a], "-jmpbit") == 0) {
        a++;
        prejmpbits = string(argv[a]);
        printf("New JMPbit %s \n", argv[a]);
        a++;
    }else if (strcmp(argv[a], "-jmpmult") == 0) {
        a++;
        multiplerJMPstr = string(argv[a]);
        printf("New JMPmult %s \n", argv[a]);
        a++;
    }

    else if(a == argc - 1) {
      configFile = string(argv[a]);
      a++;
    } else {
      printf("Unexpected %s argument\n",argv[a]);
      exit(-1);
    }

  }

  // Init SecpK1
  Secp256K1* secp = new Secp256K1();
  secp->Init(Gx,Gy);

  if(gridSize.size() == 0) {
    for(int i = 0; i < gpuId.size(); i++) {
      gridSize.push_back(0);
      gridSize.push_back(0);
    }
  } else if(gridSize.size() != gpuId.size() * 2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  Kangaroo *v = new Kangaroo(secp,dp,gpuEnable,workFile,iWorkFile,savePeriod,saveKangaroo,
                             maxStep,wtimeout,port,ntimeout,serverIP,outputFile,splitWorkFile,multG,prejmpbits, multiplerJMPstr);
  if(checkFlag) {
    v->Check(gpuId,gridSize);  
    exit(0);
  } else {
    if(infoFile.length()>0) {
      v->WorkInfo(infoFile);
      exit(0);
    } else if(merge1.length()>0) {
      v->MergeWork(merge1,merge2,mergeDest);
      exit(0);
    } if(iWorkFile.length()>0) {
      if( !v->LoadWork(iWorkFile) )
        exit(-1);
    } else if(configFile.length()>0) {
      if( !v->ParseConfigFile(configFile) )
        exit(-1);
    } else {
      if(serverIP.length()==0) {
        ::printf("No input file to process\n");
        exit(-1);
      }
    }
    if(serverMode)
      v->RunServer();
    else
      v->Run(nbCPUThread,gpuId,gridSize);
  }

  return 0;

}
