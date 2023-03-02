# want to have an input of a MAC address, then return some number based on the "level" of classification it is
# proposed:
# 0 - hard fail
# 1 - probably bad
# 2 - neutral
# 3 - probably good

def classifyMAC(macAddress: bytes):
    if len(macAddress) != 6:
        raise ValueError("MAC addresses must be only 6 bytes long!")
    
    if   macAddress == bytes.fromhex('FFFFFFFFFFFF'):
        return 3
    elif macAddress == bytes.fromhex('000000000000'):
        return 3
    elif macAddress[0:3] == bytes.fromhex('b827eb'):
        return 2
    # if the mac address is in this list of 'valid' mac addresses, then classification = 1    
    else:
        mac_lookup_file = '/Users/john/Documents/PPP/mac_lookup.txt'
        with open(mac_lookup_file, 'r', encoding='utf-8') as mac_lookup:
            # skip the first two lines in the file, since the first two 
            # lines are :
            # '# Created using PPP/functions/make_lookups.py'
            # and
            # 'Assignment Organization Name'
            mac_lookup.__next__()
            mac_lookup.__next__()
            for row in mac_lookup:
                # splits each row into a list of two values, first 
                # value is the address block assigned, and the second 
                # vlaue is the organization that owns the address block
                # ex: ['0001C7', 'Cisco Systems, Inc\n'] (type:list)
                row = row.split(' ', 1)

                # length of address block assigned to organization
                # ex: 6 (type: int) (this can only be 6, 7, or 9)
                assigned_length = len(row[0])
                if macAddress.hex().upper()[:assigned_length] == row[0]:
                    return 1
    
    # default to returning zero            
    return 0