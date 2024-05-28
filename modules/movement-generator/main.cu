#include "server.hpp"
#include "synthesiser.h"
#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "shot-file.h"
#include <string>

int main() {

	Server server{};
	AWG awg{};
	std::string hdf_address{};

	{   // listen for the handshake signal
		std::string message;	
		server.listen(message);
		if (message == "hello") {
			server.send("hello");	
		} else return 1;
	}

	
	while(true) {  // listen for hd5 address
		std::string message;	
		server.listen(message);
		if (message.substr(message.size() - 3) == ".h5") {
			hdf_address = adjust_address(message);
			break;	
		} else if (message == "done") {
			server.send("ok");	
			return 2;
		}  
	}

	{   // listen for the done signal
		std::string message;	
		server.listen(message);
		if (message == "done") {
			server.send("ok");	
		} else return 1;
	}

    ShotFile shotfile(hdf_address);
    MovementsConfig movementsConfig(shotfile);
	Synthesiser synthesiser{COEF_X_PATH("21_traps.csv"), COEF_Y_PATH("21_traps.csv"), movementsConfig};
    synthesiser.synthesise_and_upload(awg);	
	awg.start_stream();

	{   // listen for the done signal
		std::string message;	
		while (true) {
			server.listen(message);
			if (message == "done") {
				server.send("ok");	
				return 0;
			} 
		}
	}


}
