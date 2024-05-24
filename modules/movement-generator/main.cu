#include "server.hpp"
#include "awg.hpp"
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
			hdf_address = adjest_address(message);
			break;	
		} else if (message == "done") {
			server.send("ok");	
			return 2;
		}  
	}

	{   // listen for the done signal
		std::string message;	
		if (message == "done") {
		server.listen(message);
			server.send("ok");	
		} else return 1;
	}


	// READ HDF5 INTO A VECTOR OF PAIRS OF HASHMAP + BOOL (for trigger)


	// FUNC TO TRANSLATE THE HASHMAP TO CPP STRUCT

	// SYNTHESIS AND UPLOAD TO AWG

	// START STREAM

	// LISTEN FOR SERVER SIGNAL

}
