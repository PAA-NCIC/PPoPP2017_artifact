#!/usr/bin/awk -f
{
    if (NF >= 5) {
        $1=""
        print $0
    }
}
