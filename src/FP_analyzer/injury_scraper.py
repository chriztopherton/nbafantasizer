"""
ESPN and CBS Sports Injury Report Scraper
Scrapes injury information from ESPN's NBA injuries page and CBS Sports
to provide enhanced injury reports with team information and updated dates
"""

import re

import requests
from bs4 import BeautifulSoup


class ESPNInjuryScraper:
    """Scraper for ESPN and CBS Sports NBA injury reports"""

    BASE_URL = "https://www.espn.com/nba/injuries"
    CBS_BASE_URL = "https://www.cbssports.com/nba/injuries/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._injury_cache: dict[str, dict] | None = None
        self._cbs_injury_cache: dict[str, dict] | None = None

    def _normalize_name(self, name: str) -> str:
        """Normalize player name for matching"""
        # Remove common suffixes and normalize whitespace
        name = re.sub(r"\s+", " ", name.strip())
        # Remove common suffixes like Jr., Sr., III, etc.
        name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "", name, flags=re.IGNORECASE)
        return name.lower()

    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy matching)"""
        name1_norm = self._normalize_name(name1)
        name2_norm = self._normalize_name(name2)

        # Exact match after normalization
        if name1_norm == name2_norm:
            return True

        # Check if one name contains the other (handles nicknames)
        if name1_norm in name2_norm or name2_norm in name1_norm:
            return True

        # Split into parts and check if last name matches
        parts1 = name1_norm.split()
        parts2 = name2_norm.split()
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Check if last names match
            if parts1[-1] == parts2[-1]:
                # Check if first names are similar (first letter match)
                if parts1[0][0] == parts2[0][0]:
                    return True

        return False

    def fetch_injuries(self) -> dict[str, dict]:
        """
        Fetch all injuries from ESPN and return as a dictionary
        Key: normalized player name, Value: injury info dict
        """
        if self._injury_cache is not None:
            return self._injury_cache

        try:
            response = self.session.get(self.BASE_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            injuries = {}

            # ESPN uses ResponsiveTable class for injury tables
            # Look for tables with ResponsiveTable class
            injury_tables = soup.find_all("div", class_=re.compile(r"ResponsiveTable", re.I))

            if not injury_tables:
                # Fallback: look for any table element
                injury_tables = soup.find_all("table")

            for table in injury_tables:
                # Look for table rows (tr) or div rows within the table
                rows = table.find_all(
                    ["tr", "div"], class_=re.compile(r"Table__TR|Table__Row|Row", re.I)
                )
                if not rows:
                    # If no specific row class, get all tr elements
                    rows = table.find_all("tr")

                for row in rows:
                    player_info = self._extract_player_injury_from_row(row)
                    if player_info:
                        normalized_name = self._normalize_name(player_info["name"])
                        injuries[normalized_name] = player_info
                        # Also store with original name for exact matching
                        injuries[player_info["name"].lower()] = player_info

            # If we didn't find structured data, try parsing text content
            if not injuries:
                injuries = self._parse_text_based_injuries(soup)

            self._injury_cache = injuries
            return injuries

        except requests.RequestException as e:
            print(f"Error fetching injuries: {e}")
            return {}
        except Exception as e:
            print(f"Error parsing injuries: {e}")
            return {}

    def fetch_cbs_injuries(self) -> dict[str, dict]:
        """
        Fetch all injuries from CBS Sports and return as a dictionary
        Key: normalized player name, Value: injury info dict with team and updated date
        """
        if self._cbs_injury_cache is not None:
            return self._cbs_injury_cache

        try:
            response = self.session.get(self.CBS_BASE_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            injuries = {}

            # CBS Sports uses TableBaseWrapper class for team injury tables
            team_wrappers = soup.find_all("div", class_="TableBaseWrapper")

            for team_wrapper in team_wrappers:
                try:
                    # Extract team name
                    team_name_elem = team_wrapper.find("div", class_="TeamLogoNameLockup-name")
                    team_name = team_name_elem.get_text(strip=True) if team_name_elem else ""

                    # Find all player rows
                    player_rows = team_wrapper.find_all("tr", class_="TableBase-bodyTr")

                    for player_row in player_rows:
                        try:
                            cells = player_row.find_all("td", class_="TableBase-bodyTd")
                            if len(cells) < 5:
                                continue

                            # Extract player name
                            name_elem = player_row.find("span", class_="CellPlayerName--long")
                            if not name_elem:
                                continue
                            name = name_elem.get_text(strip=True)

                            if not name or len(name.split()) < 2:
                                continue

                            # Extract position
                            position = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                            # Extract updated date
                            updated_elem = player_row.find("span", class_="CellGameDate")
                            updated_date = updated_elem.get_text(strip=True) if updated_elem else ""

                            # Extract injury description
                            injury_desc = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                            # Extract status
                            status = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                            # Create injury info dict
                            normalized_name = self._normalize_name(name)
                            injury_info = {
                                "name": name,
                                "team": team_name,
                                "position": position,
                                "updated": updated_date,
                                "injury": injury_desc,
                                "status": status,
                            }

                            # Store with normalized name
                            injuries[normalized_name] = injury_info
                            # Also store with original name for exact matching
                            injuries[name.lower()] = injury_info

                        except Exception as e:
                            print(f"Error extracting player from CBS row: {e}")
                            continue

                except Exception as e:
                    print(f"Error processing CBS team wrapper: {e}")
                    continue

            self._cbs_injury_cache = injuries
            return injuries

        except requests.RequestException as e:
            print(f"Error fetching CBS injuries: {e}")
            return {}
        except Exception as e:
            print(f"Error parsing CBS injuries: {e}")
            return {}

    def _extract_player_injury_from_row(self, row) -> dict | None:
        """Extract player injury info from a table row"""
        try:
            # ESPN typically structures rows with Table__TD or td elements
            cells = row.find_all(["td", "div"], class_=re.compile(r"Table__TD|TableCell", re.I))
            if not cells:
                cells = row.find_all("td")

            if len(cells) < 2:  # Need at least name and status
                return None

            # First cell usually contains player name and position
            name_cell = cells[0]
            name = ""
            position = ""

            # Look for anchor tag (ESPN often links player names)
            name_link = name_cell.find("a")
            if name_link:
                name = name_link.get_text(strip=True)
            else:
                # Try to extract name from text
                name_text = name_cell.get_text(strip=True)
                # Name is usually first, position might be in parentheses or after
                name_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", name_text)
                if name_match:
                    name = name_match.group(1)
                    # Extract position if present (usually in parentheses or after name)
                    pos_match = re.search(r"\(([A-Z]+)\)|([A-Z]+)$", name_text)
                    if pos_match:
                        position = pos_match.group(1) or pos_match.group(2)

            if not name or len(name.split()) < 2:
                return None

            # ESPN structure: typically has 5 columns
            # Column 1 (col-name): Player name
            # Column 2 (col-pos): Position
            # Column 3 (col-date): Return date
            # Column 4 (col-stat): Status (Out, Day-to-Day, etc.)
            # Column 5 (col-desc): Comment/Description

            status = ""
            return_date = ""
            description = ""
            comment = ""

            # Try to extract position from name cell if not already found
            if not position:
                name_text = name_cell.get_text(strip=True)
                # Position might be after name, separated by space or in parentheses
                pos_match = re.search(r"\(([A-Z]+)\)|([A-Z]+)$", name_text)
                if pos_match:
                    position = pos_match.group(1) or pos_match.group(2)

            # Look for cells by class to identify their purpose
            for cell in cells:
                cell_classes = cell.get("class", [])
                cell_text = cell.get_text(strip=True)

                # Convert cell_classes to a set for easier checking (handle both list and string)
                class_set = set(cell_classes) if isinstance(cell_classes, list) else {cell_classes}

                # Position cell (col-pos)
                if "col-pos" in class_set or (
                    not position
                    and len(cell_text) <= 3
                    and cell_text in ["F", "C", "G", "PG", "SG", "SF", "PF"]
                ):
                    if not position:
                        position = cell_text

                # Date cell (col-date)
                elif "col-date" in class_set:
                    return_date = cell_text

                # Status cell (col-stat)
                elif "col-stat" in class_set:
                    status_text = cell_text
                    # Status might be mixed with other info, try to extract
                    # Common statuses: Out, Day-to-Day, Questionable, Probable, etc.
                    status_match = re.search(
                        r"(Out|Day-to-Day|Questionable|Probable|Doubtful|Injured)",
                        status_text,
                        re.I,
                    )
                    if status_match:
                        status = status_match.group(1)
                    else:
                        status = status_text

                # Comment/Description cell (col-desc) - This is the most important for full comment
                elif "col-desc" in class_set:
                    comment = cell_text
                    description = cell_text  # Also store as description for backwards compatibility

            # Fallback: if we didn't find cells by class, use positional logic
            if not status and len(cells) > 1:
                # Try second cell for status
                status_text = cells[1].get_text(strip=True)
                status_match = re.search(
                    r"(Out|Day-to-Day|Questionable|Probable|Doubtful|Injured)", status_text, re.I
                )
                if status_match:
                    status = status_match.group(1)
                elif status_text not in ["F", "C", "G", "PG", "SG", "SF", "PF"]:  # Not a position
                    status = status_text

            if not return_date and len(cells) > 2:
                date_text = cells[2].get_text(strip=True)
                # Try to extract return date
                date_match = re.search(
                    r"(?:until|out until|return|expected)\s+([A-Z][a-z]+\s+\d+)|([A-Z][a-z]+\s+\d+)",
                    date_text,
                    re.I,
                )
                if date_match:
                    return_date = date_match.group(1) or date_match.group(2)
                elif not any(
                    x in date_text.lower() for x in ["out", "day", "questionable"]
                ):  # Not a status
                    return_date = date_text

            if not comment and len(cells) > 3:
                desc_text = cells[3].get_text(strip=True)
                if desc_text and len(desc_text) > 10:  # Likely a comment if it's longer
                    comment = desc_text
                    description = desc_text

            # If we still don't have a good status, try to infer from description
            if not status or status in [
                "F",
                "C",
                "G",
                "PG",
                "SG",
                "SF",
                "PF",
            ]:  # These are positions, not statuses
                if description:
                    status_match = re.search(
                        r"(Out|Day-to-Day|Questionable|Probable|Doubtful|Injured)",
                        description,
                        re.I,
                    )
                    if status_match:
                        status = status_match.group(1)
                    elif "out" in description.lower():
                        status = "Out"
                    elif "day" in description.lower():
                        status = "Day-to-Day"
                    else:
                        status = "Injured"

            return {
                "name": name,
                "status": status or "Unknown",
                "description": description,
                "comment": comment,
                "return_date": return_date,
                "position": position,
            }

        except Exception as e:
            print(f"Error extracting from row: {e}")
            return None

    def _parse_text_based_injuries(self, soup: BeautifulSoup) -> dict[str, dict]:
        """Fallback: Parse injuries from text content if structured parsing fails"""
        injuries = {}

        # Look for common ESPN injury page patterns
        # ESPN often uses specific data structures or JSON embedded in the page
        scripts = soup.find_all("script", type="application/json")
        for script in scripts:
            try:
                import json

                data = json.loads(script.string)
                # Recursively search for player injury data
                injuries.update(self._extract_from_json(data))
            except (json.JSONDecodeError, AttributeError):
                continue

        # If JSON parsing didn't work, try finding all text that mentions injuries
        # This is a last resort and may be less reliable
        if not injuries:
            # Look for patterns like "Player Name - Status - Description"
            text_content = soup.get_text()
            # This is a simplified pattern - may need adjustment based on actual page structure
            pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–]\s*([^-\n]+)"
            matches = re.finditer(pattern, text_content)
            for match in matches:
                name = match.group(1).strip()
                info = match.group(2).strip()
                if len(name.split()) >= 2:  # Valid name
                    normalized_name = self._normalize_name(name)
                    injuries[normalized_name] = {
                        "name": name,
                        "status": info.split("-")[0].strip() if "-" in info else "Unknown",
                        "description": info,
                        "comment": info,
                        "return_date": "",
                        "position": "",
                    }

        return injuries

    def _extract_from_json(self, data, depth=0) -> dict[str, dict]:
        """Recursively extract injury data from JSON structure"""
        injuries = {}

        if depth > 10:  # Prevent infinite recursion
            return injuries

        if isinstance(data, dict):
            # Look for common keys that might contain player/injury data
            if "name" in data and "status" in data:
                name = data.get("name", "")
                if name and len(name.split()) >= 2:
                    normalized_name = self._normalize_name(name)
                    injuries[normalized_name] = {
                        "name": name,
                        "status": data.get("status", "Unknown"),
                        "description": data.get("description", data.get("injury", "")),
                        "comment": data.get(
                            "comment", data.get("description", data.get("injury", ""))
                        ),
                        "return_date": data.get("returnDate", data.get("return_date", "")),
                        "position": data.get("position", ""),
                    }

            # Recursively search nested structures
            for value in data.values():
                injuries.update(self._extract_from_json(value, depth + 1))

        elif isinstance(data, list):
            for item in data:
                injuries.update(self._extract_from_json(item, depth + 1))

        return injuries

    def get_player_injury(self, player_name: str) -> dict | None:
        """
        Get injury information for a specific player from both ESPN and CBS Sports

        Args:
            player_name: Player name (e.g., "LeBron James")

        Returns:
            Dictionary with enhanced injury info or None if not found/not injured
            Enhanced fields include: team, updated (from CBS Sports)
        """
        # Fetch from both sources
        espn_injuries = self.fetch_injuries()
        cbs_injuries = self.fetch_cbs_injuries()

        # Try to find player in ESPN injuries
        espn_injury = None
        if espn_injuries:
            # Try exact match first (case-insensitive)
            player_name_lower = player_name.lower()
            if player_name_lower in espn_injuries:
                espn_injury = espn_injuries[player_name_lower]
            else:
                # Try normalized match
                normalized_name = self._normalize_name(player_name)
                if normalized_name in espn_injuries:
                    espn_injury = espn_injuries[normalized_name]
                else:
                    # Try fuzzy matching
                    for _injury_name, injury_data in espn_injuries.items():
                        if self._fuzzy_match(player_name, injury_data["name"]):
                            espn_injury = injury_data
                            break

        # Try to find player in CBS Sports injuries
        cbs_injury = None
        if cbs_injuries:
            player_name_lower = player_name.lower()
            normalized_name = self._normalize_name(player_name)

            if player_name_lower in cbs_injuries:
                cbs_injury = cbs_injuries[player_name_lower]
            elif normalized_name in cbs_injuries:
                cbs_injury = cbs_injuries[normalized_name]
            else:
                # Try fuzzy matching
                for _injury_name, injury_data in cbs_injuries.items():
                    if self._fuzzy_match(player_name, injury_data["name"]):
                        cbs_injury = injury_data
                        break

        # If no injury found in either source, return None
        if not espn_injury and not cbs_injury:
            return None

        # Merge data from both sources, prioritizing ESPN data
        merged_injury = {}

        # Start with ESPN data if available
        if espn_injury:
            merged_injury = espn_injury.copy()
        else:
            # If only CBS data available, create structure compatible with ESPN format
            merged_injury = {
                "name": cbs_injury.get("name", player_name),
                "status": cbs_injury.get("status", "Unknown"),
                "description": cbs_injury.get("injury", ""),
                "comment": cbs_injury.get("injury", ""),
                "return_date": "",
                "position": cbs_injury.get("position", ""),
            }

        # Enhance with CBS Sports data
        if cbs_injury:
            # Add team information
            if "team" in cbs_injury and cbs_injury["team"]:
                merged_injury["team"] = cbs_injury["team"]

            # Add updated date
            if "updated" in cbs_injury and cbs_injury["updated"]:
                merged_injury["updated"] = cbs_injury["updated"]

            # Enhance description if CBS has more detailed info
            cbs_injury_desc = cbs_injury.get("injury", "")
            if cbs_injury_desc and (
                not merged_injury.get("description")
                or len(cbs_injury_desc) > len(merged_injury.get("description", ""))
            ):
                merged_injury["description"] = cbs_injury_desc
                merged_injury["comment"] = cbs_injury_desc

            # Enhance status if CBS has more specific status
            cbs_status = cbs_injury.get("status", "")
            if cbs_status and (
                not merged_injury.get("status") or merged_injury.get("status") == "Unknown"
            ):
                merged_injury["status"] = cbs_status

            # Enhance position if missing
            if not merged_injury.get("position") and cbs_injury.get("position"):
                merged_injury["position"] = cbs_injury["position"]

        return merged_injury

    def get_all_injuries(self) -> list[dict]:
        """Get list of all injured players from both ESPN and CBS Sports"""
        espn_injuries = self.fetch_injuries()
        cbs_injuries = self.fetch_cbs_injuries()

        # Combine all unique players
        all_players = set()
        result = []

        # Add ESPN injuries
        for injury_data in espn_injuries.values():
            name_lower = injury_data["name"].lower()
            if name_lower not in all_players:
                all_players.add(name_lower)
                result.append(injury_data)

        # Add CBS injuries that aren't already in the list
        for injury_data in cbs_injuries.values():
            name_lower = injury_data["name"].lower()
            if name_lower not in all_players:
                all_players.add(name_lower)
                # Convert CBS format to standard format
                result.append(
                    {
                        "name": injury_data.get("name", ""),
                        "status": injury_data.get("status", "Unknown"),
                        "description": injury_data.get("injury", ""),
                        "comment": injury_data.get("injury", ""),
                        "return_date": "",
                        "position": injury_data.get("position", ""),
                        "team": injury_data.get("team", ""),
                        "updated": injury_data.get("updated", ""),
                    }
                )
            else:
                # Enhance existing injury with CBS data
                for existing_injury in result:
                    if existing_injury["name"].lower() == name_lower:
                        if "team" not in existing_injury and injury_data.get("team"):
                            existing_injury["team"] = injury_data["team"]
                        if "updated" not in existing_injury and injury_data.get("updated"):
                            existing_injury["updated"] = injury_data["updated"]
                        break

        return result

    def clear_cache(self):
        """Clear both injury caches to force a fresh fetch"""
        self._injury_cache = None
        self._cbs_injury_cache = None
