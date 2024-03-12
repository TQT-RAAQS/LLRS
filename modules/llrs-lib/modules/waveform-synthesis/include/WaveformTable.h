#ifndef _WAVEFORMTABLE_H_
#define _WAVEFORMTABLE_H_

#include "WaveformRepo.h"
#include "LLRS-lib/Settings.h"
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <sys/mman.h> 
#include <stdexcept> 


namespace Synthesis{

    /**
     * @brief The Enum representing the type of moves using waveforms
     * The reason for having duplicate enums for 1d and 2d is made purely out of caution and for the sake of readibility.  
     */
    enum WfMoveType {
        IDLE_1D,      // 0
        IMPLANT_1D,   // 1
        EXTRACT_1D,   // 2
        FORWARD_1D,   // 3
        BACKWARD_1D,  // 4
        IDLE_2D,      // 5
        IMPLANT_2D,   // 6
        EXTRACT_2D,   // 7  
        RIGHT_2D,     // 8
        LEFT_2D,      // 9 
        UP_2D,        // 10
        DOWN_2D,      // 11
    };
    
    /**
     * @brief A Waveform page consisting of a vector of Waveforms
     * 
     */
    using WF_PAGE = std::vector<std::vector<short>>;

    /**
     * @brief A Table page consisting of a page of waveforms and data related to it.
     *   
     */
    using TABLE_PAGE = std::tuple<WfType, WfType, size_t, WF_PAGE>;

    /**
     * @brief The tuple of move type and extraction extent used as the hashmap keys.
     */
    using Table_key = std::pair<WfMoveType, int>;

    /**
    * @brief sets a bit of each discretized waveform sample of a to 0 according to WFM_MASK
    * as well as converts sample type from double to short
    * @param src => vector (waveform) for which values (samples) to translate
    */
    std::vector<short> translate_waveform(const std::vector<double> &src);


// ---------------------------------------------------------------------------------------------

    /** 
     * @class WaveformTable
     * @brief the WaveformTable class interleaves waveforms from a WaveformRepo and puts them in
     * arrays to be used when translating RedRec moves to waveforms
    */
    class WaveformTable{
        /// The Pages are saved in primary fields in 1D and in the base_table in 2D.
        Channel                 primary_chan;
        Channel                 secondary_chan;
        size_t                  primary_size;
        size_t                  secondary_size;
        WF_PAGE                 primary_static;
        WF_PAGE                 primary_extract;
        WF_PAGE                 primary_implant;
        std::vector<WF_PAGE>    primary_forward;
        std::vector<WF_PAGE>    primary_backward;
        WF_PAGE                 secondary_static;
        WF_PAGE                 secondary_extract;
        WF_PAGE                 secondary_implant;
        WF_PAGE                 secondary_leftward;
        WF_PAGE                 secondary_rightward;
        std::map<Table_key, TABLE_PAGE> base_table;

    public:
        /**
        * @brief Constructs all of the ingedients to form 
        *   interleaved operations based on algorithm
        * @param p_repo => points to a Waveform Repo that contains
        *    all of the waveforms from each channel
        * @param is_transposed => boolean that indicates primary 
        *   and secoondary channel values, by default we say CHAN1 is primary(column-wise)
        *   and CHAN0 is secondary (row-wise)
        */
        WaveformTable(WaveformRepo * p_repo, bool is_transposed);
        
        /**
         * @brief The default constructor
        */
        WaveformTable() = default;

        /**
         * @brief Get a pointer to a specified waveform 
         * 
         * @param move  The Enum corresponding to the Wf's move
         * @param extraction_extent 
         * @param index 
         * @param offset 
         * @param block_size 
         * @return short* Pointer to the waveform 
         */
        short * get_waveform_ptr(WfMoveType move, int extraction_extent, int index, int offset, int block_size);

    /// Only used in testing 

        /**
        * @brief sample wise interleave: even samples are for CHAN0 and odd are for CHAN1
        * @return interleaved wf double the size of the repo's waveform length
        */
        std::vector<short> interleave_waveforms(std::vector<short> even_wf, std::vector<short> odd_wf);

        /**
        * @brief addressing logic for primary waveforms:
        *   This ensures continuous addresses for index + blocksize < size
        *   Note that displacement are made s.t. index + blocksize < size - 1
        */
        size_t get_blocked_addr(WfType wf_type, size_t index, size_t block_size, size_t max);

        /**
        * @brief Addressing into any primary ingredients
        */
        short * get_primary_wf_ptr(WfType wf_type, size_t index, size_t block_size, size_t extraction_extent);

        /**
        * @brief Addressing into any secondary ingredients
        */
        std::vector<short> get_secondary_wf(WfType wf_type, size_t index);

        /**
        * @brief Helper to use pointers to ingredients
        */
        WF_PAGE * get_primary_pointer(WfType wf_type, size_t max);

        /**
        * @brief Helper to use pointers to ingredients
        */
        WF_PAGE * get_secondary_pointer(WfType wf_type);
 
        /**
        * @brief Used only in constructor to create primary ingredients
        */
        WF_PAGE init_primary(WfType wf_type, size_t range, WaveformRepo * p_repo);

        /**
        * @brief Used only in constructor to create secondary ingredients
        */
        WF_PAGE init_secondary(WfType wf_type, size_t range, WaveformRepo * p_repo);

        /**
        * @brief Create needed pages into member table
        */
        void init_table();
 
        /**
        * @brief Merge two ingredients into a new "page" that contains 
        *   interleaved waveforms from a primary and secondary.
        */
        WF_PAGE merge_pages(WF_PAGE * wf_type_even, WF_PAGE * wf_type_odd);
    };
    
}

#endif
