#ifndef _WAVEFORMREPO_H_
#define _WAVEFORMREPO_H_

#include "Waveform.h"
#include "LLRS-lib/Settings.h"
#include "WaveformPowerCalculator.h"
#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>
#include <cassert>
#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>



namespace Synthesis {
    

    /** 
    * @brief turns a human-parsable repo key spec to its unique uint32_t representation
    * @param channel => AWG channel to stream to (CHAN_0 [X] or CHAN_1 [Y])
    * @param type => waveform type 
    * @param index => index of leftmost atom in block that is being targeted (0 to [_n_x-1 or _n_y-1])
    * @param block_size => size of block of targeted atoms (1 to [_n_x or _n_y])
    * @param extraction_extent => how many traps are being extracted (0 to [_n_x-1 or _n_y-1]) - this dictates how many static waveforms will be added
    */
    std::uint32_t get_key(Channel channel, WfType type, std::uint32_t index, std::uint32_t block_size, std::uint32_t extraction_extent);


    /**
    * @brief Adds the value at each index of src vector to corresponding
    *      value of dst vector without copies. not to be confused with concatenating 2 vectors
    * @param src => vector that values are taken from
    * @param dst => vector that the src values are added to
    */
    void element_wise_add(std::vector<double> src, std::vector<double>& dst);

    /**
     * @class WaveformRepo
     * @brief WaveformRepo Class is used to generate all possible discretized block RF waveforms into a repository (hashmap datastructure) for efficient lookup
    */
    class WaveformRepo {
        std::size_t _n_x; 
        std::size_t _n_y;
        double _sample_rate;
        double _wf_duration;
        size_t _wf_len;
        std::unordered_map<std::uint32_t, std::vector<double>>  _waveform_hashmap;
       
    public:
        /**
        * @brief waveform repo class constructor
        * @param _n_x => horizontal dimension of repo (corresponds to CHAN_0 keys)
        * @param _n_y => vertical dimension of repo (corresponds to CHAN_1 keys)
        * @param sample_rate
        * @param wf_duration
        */
        WaveformRepo(std::size_t n_x, std::size_t n_y, double sample_rate, double wf_duration);
        

        /**
        * @brief generates all needed waveforms according to repo dimensions,
        *           and stores them in waveform hashmap.  
        * @param params_x, params_y => vector of optimization/frequency coefficients for each index of 
        * @param df difference between frequnecies corresponding to consecutive columns/rows
        */ 
        void generate_repository(std::vector<WP> params_x, std::vector<WP>  params_y, double df = 0.5e6);

 
        /**
        * @brief returns the waveform repo entry corresponing to a provided waveform
        * @param channel => AWG channel to stream to (CHAN_0 [X] or CHAN_1 [Y])
        * @param waveform_type => waveform type 
        * @param index => index of leftmost atom in block that is being targeted (0 to (_n_x-1 or _n_y-1))
        * @param block_size => size of block of targeted atoms
        */
        std::vector<double> get_waveform(Channel channel, WfType type, std::uint32_t index, std::uint32_t block_size, std::uint32_t extraction_extent) const;

        /**
        * @brief returns the waveform repo entry corresponing to a human-parsable repo key spec
        * @param key => uint32_t corresponding to a particular waveform
        */
        std::vector<double> get_waveform(std::uint32_t key) const;


        /**
        * @brief caches (saves) waveform repo to file with specified file name, in binary
        *           the format of a saved waveform is:
        *           [key][sample 1][sample 2]...[sample _wf_len]
        *           all in binary. note the lack of spaces/newlines.
        * @param  fname => name of file to save waveform repo to
        * @return  LLRS_OK on success, LLRS_ERR on error */
        int cache_waveform_repo(std::string filename="") const;

        /**
        * @brief loads a waveform repo from binary file with specified file name
        * @param fname => name of binary file to load waveform repo from
        * @return  LLRS_OK on success, LLRS_ERR on error
        */
        int load_waveform_repo(std::string filename);

        /**
        * @brief caches (saves) waveform repo to file with specified file name, in plaintext
        *           this format makes each key and is corresponding vector of shorts readable.
        *           the format of a saved waveform is:
        *           "[key] [number of samples] [sample 1] [sample 2] ... [sample _wf_len]\n"
        * @param fname => name of file to save waveform repo to
        * @return LLRS_OK on success, LLRS_ERR on error
        */
        int cache_waveform_repo_plaintext(std::string filename="") const;

        /**
        * @brief loads a waveform repo from plaintext file with specified file name
        * @param fname => name of plaintext file to load waveform repo from
        * @return LLRS_OK on success, LLRS_ERR on error
        */
        int load_waveform_repo_plaintext(std::string filename);


        /**
         * @brief Getter for the raw hashmap
         * @return Raw Hashmap 
         */ 
        std::unordered_map<std::uint32_t, std::vector<double> > get_raw_map () const;
        /**
         * @brief Getter for the size of the repo
         * @return Repo Size 
         */ 
        std::int32_t get_repo_size();
        /**
         * @brief Getter for the x dimension of the repo
         * @return X-dimension 
         */ 
        std::size_t get_dimension_x();
        /**
         * @brief Getter for the y dimension of the repo
         * @return Y-dimension 
         */ 
        std::size_t get_dimension_y();
 
        /**
        * @brief equals operator used to compare 2 waveform repos, used for testing
        * @param other => waveform repo to compare with
        */
        bool operator==(const WaveformRepo &other) const;

        /**
        * @brief not-equals operator used to compare 2 waveform repos, used for testing
        * @param other => waveform repo to compare with
        */
        bool operator!=(const WaveformRepo &other) const;
    
    private:  
        /**
        * @brief generates all static waveforms (both monochromatic and block waveforms)  
        * @param channel => which channel (axis) to create waveforms for (CHAN_0 or CHAN_1)
        * @param n => size of atom array
        * @param params => vector of optimization/frequency coefficients for each index of atom array 
        */
        void generate_static_waveforms  (Channel channel, std::vector<WP> params, std::size_t n );

        /**
        * @brief generates all displacement waveforms (monochromatic and block waveforms 
        *           for forwards and backwards displacement)  
        * @param channel => which channel (axis) to create waveforms for (CHAN_0 or CHAN_1)
        * @param n => size of atom array
        * @param params => vector of optimization/frequency coefficients for each index of atom array  
        */
        void generate_displacement_waveforms(Channel channel, std::vector<WP> params, std::size_t n, double df);

        /**
        * @brief generates all transfer waveforms (monochromatic and block waveforms for extraction and implantation)  
        * @param channel => which channel (axis) to create waveforms for (CHAN_0 or CHAN_1)
        * @param n => size of atom array
        * @param params => vector of optimization/frequency coefficients for each index of atom array 
        */
        void generate_transfer_waveforms(Channel channel, std::vector<WP> params, std::size_t n );


    };

}

#endif
